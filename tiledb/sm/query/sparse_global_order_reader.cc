/**
 * @file   sparse_global_order_reader.cc
 *
 * @section LICENSE
 *
 * The MIT License
 *
 * @copyright Copyright (c) 2017-2021 TileDB, Inc.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 * THE SOFTWARE.
 *
 * @section DESCRIPTION
 *
 * This file implements class SparseGlobalOrderReader.
 */

#include "tiledb/sm/query/sparse_global_order_reader.h"
#include "tiledb/common/logger.h"
#include "tiledb/sm/array/array.h"
#include "tiledb/sm/array_schema/array_schema.h"
#include "tiledb/sm/array_schema/dimension.h"
#include "tiledb/sm/fragment/fragment_metadata.h"
#include "tiledb/sm/misc/comparators.h"
#include "tiledb/sm/misc/hilbert.h"
#include "tiledb/sm/misc/parallel_functions.h"
#include "tiledb/sm/misc/utils.h"
#include "tiledb/sm/query/query_macros.h"
#include "tiledb/sm/query/result_tile.h"
#include "tiledb/sm/stats/global_stats.h"
#include "tiledb/sm/storage_manager/open_array_memory_tracker.h"
#include "tiledb/sm/storage_manager/storage_manager.h"
#include "tiledb/sm/subarray/subarray.h"

using namespace tiledb;
using namespace tiledb::common;
using namespace tiledb::sm::stats;

namespace tiledb {
namespace sm {

/* ****************************** */
/*          CONSTRUCTORS          */
/* ****************************** */

SparseGlobalOrderReader::SparseGlobalOrderReader(
    stats::Stats* stats,
    tdb_shared_ptr<Logger> logger,
    StorageManager* storage_manager,
    Array* array,
    Config& config,
    std::unordered_map<std::string, QueryBuffer>& buffers,
    Subarray& subarray,
    Layout layout,
    QueryCondition& condition)
    : SparseIndexReaderBase(
          stats,
          logger->clone("SparseGlobalOrderReader", ++logger_id_),
          storage_manager,
          array,
          config,
          buffers,
          subarray,
          layout,
          condition)
    , empty_result_tiles_(true) {
  array_memory_tracker_ =
      storage_manager_->array_memory_tracker(array->array_uri());
}

/* ****************************** */
/*               API              */
/* ****************************** */

bool SparseGlobalOrderReader::incomplete() const {
  return !read_state_.done_adding_result_tiles_ || !empty_result_tiles_;
}

Status SparseGlobalOrderReader::init() {
  RETURN_NOT_OK(SparseIndexReaderBase::init());

  // Initialize memory budget variables.
  RETURN_NOT_OK(initialize_memory_budget());

  return Status::Ok();
}

Status SparseGlobalOrderReader::initialize_memory_budget() {
  bool found = false;
  RETURN_NOT_OK(
      config_.get<uint64_t>("sm.mem.total_budget", &memory_budget_, &found));
  assert(found);
  RETURN_NOT_OK(config_.get<double>(
      "sm.mem.reader.sparse_global_order.ratio_array_data",
      &memory_budget_ratio_array_data_,
      &found));
  assert(found);
  RETURN_NOT_OK(config_.get<double>(
      "sm.mem.reader.sparse_global_order.ratio_coords",
      &memory_budget_ratio_coords_,
      &found));
  assert(found);
  RETURN_NOT_OK(config_.get<double>(
      "sm.mem.reader.sparse_global_order.ratio_query_condition",
      &memory_budget_ratio_query_condition_,
      &found));
  assert(found);
  RETURN_NOT_OK(config_.get<double>(
      "sm.mem.reader.sparse_global_order.ratio_tile_ranges",
      &memory_budget_ratio_tile_ranges_,
      &found));
  assert(found);

  return Status::Ok();
}

Status SparseGlobalOrderReader::dowork() {
  auto timer_se = stats_->start_timer("dowork");

  // For easy reference.
  auto fragment_num = fragment_metadata_.size();

  // Check that the query condition is valid.
  RETURN_NOT_OK(condition_.check(array_schema_));

  get_dim_attr_stats();

  // Handle empty array.
  if (fragment_metadata_.empty()) {
    read_state_.done_adding_result_tiles_ = true;
    empty_result_tiles_ = true;
    zero_out_buffer_sizes();
    return Status::Ok();
  }

  reset_buffer_sizes();

  // Make sure we have enough space for tiles data.
  memory_used_for_coords_.resize(fragment_num);
  memory_used_for_qc_tiles_.resize(fragment_num);
  result_tiles_.resize(fragment_num);

  // Load initial data, if not loaded already.
  RETURN_NOT_OK(load_initial_data());

  // Create the result tiles we are going to process.
  bool tiles_found = false;
  RETURN_NOT_OK(create_result_tiles(&tiles_found));

  if (tiles_found) {
    coords_loaded_ = true;

    // Maintain a temporary vector with pointers to result tiles for calling
    // read_and_unfilter_coords.
    std::vector<ResultTile*> tmp_result_tiles;
    for (auto& rt_list : result_tiles_) {
      for (auto& result_tile : rt_list) {
        tmp_result_tiles.emplace_back(&result_tile);
      }
    }

    // Read and unfilter coords.
    RETURN_NOT_OK(read_and_unfilter_coords(true, &tmp_result_tiles));

    // Compute the tile bitmaps.
    RETURN_NOT_OK(compute_tile_bitmaps<uint8_t>(&tmp_result_tiles));

    // Apply query condition.
    RETURN_NOT_OK(apply_query_condition<uint8_t>(&tmp_result_tiles));

    // Clear result tiles that are not necessary anymore.
    auto status = parallel_for(
        storage_manager_->compute_tp(), 0, fragment_num, [&](uint64_t f) {
          auto it = result_tiles_[f].begin();
          while (it != result_tiles_[f].end()) {
            if (it->bitmap_result_num_ == 0) {
              RETURN_NOT_OK(remove_result_tile(f, it++));
            } else {
              it++;
            }
          }

          return Status::Ok();
        });
    RETURN_NOT_OK_ELSE(status, logger_->status(status));

    // Compute hilbert values.
    if (array_schema_->cell_order() == Layout::HILBERT) {
      RETURN_NOT_OK(compute_hilbert_values(&tmp_result_tiles));
    }
  }

  // Compute RCS.
  std::vector<ResultCellSlab> result_cell_slabs;
  RETURN_NOT_OK(compute_result_cell_slab(&result_cell_slabs));

  // No more tiles to process, done.
  if (result_cell_slabs.empty()) {
    read_state_.done_adding_result_tiles_ = true;
    empty_result_tiles_ = true;
    zero_out_buffer_sizes();
    return Status::Ok();
  }

  // TODO Copy.

  // End the iteration.
  RETURN_NOT_OK(end_iteration());

  return Status::Ok();
}

void SparseGlobalOrderReader::reset() {
}

Status SparseGlobalOrderReader::clear_result_tiles() {
  if (!result_tiles_.empty()) {
    for (unsigned f = 0; f < fragment_metadata_.size(); f++) {
      while (!result_tiles_[f].empty()) {
        RETURN_NOT_OK(remove_result_tile(f, result_tiles_[f].begin()));
      }
    }
  }

  coords_loaded_ = false;

  return Status::Ok();
}

Status SparseGlobalOrderReader::add_result_tile(
    const unsigned dim_num,
    const uint64_t memory_budget_coords_tiles,
    const uint64_t memory_budget_qc_tiles,
    const unsigned f,
    const uint64_t t,
    const ArraySchema* const array_schema,
    bool* budget_exceeded) {
  // Calculate memory consumption for this tile.
  uint64_t tiles_size = 0, tiles_size_qc = 0;
  RETURN_NOT_OK(get_coord_tiles_size<uint8_t>(
      true, dim_num, f, t, &tiles_size, &tiles_size_qc));

  // Account for hilbert data.
  if (array_schema_->cell_order() == Layout::HILBERT) {
    tiles_size += fragment_metadata_[f]->cell_num(t) * sizeof(uint64_t);
  }

  // Don't load more tiles than the memory budget.
  if (memory_used_for_coords_[f] + tiles_size > memory_budget_coords_tiles ||
      memory_used_for_qc_tiles_[f] + tiles_size_qc > memory_budget_qc_tiles) {
    *budget_exceeded = true;
    return Status::Ok();
  }

  // Adjust total memory used.
  {
    std::unique_lock<std::mutex> lck(mem_budget_mtx_);
    memory_used_for_coords_total_ += tiles_size + sizeof(ResultTile);
    memory_used_qc_tiles_total_ += tiles_size_qc;
  }

  // Adjust per fragment memory used.
  memory_used_for_coords_[f] += tiles_size + sizeof(ResultTile);
  memory_used_for_qc_tiles_[f] += tiles_size_qc;

  // Add the tile.
  empty_result_tiles_ = false;
  result_tiles_[f].emplace_back(f, t, array_schema);

  return Status::Ok();
}

Status SparseGlobalOrderReader::create_result_tiles(bool* tiles_found) {
  auto timer_se = stats_->start_timer("create_result_tiles");

  // For easy reference.
  auto fragment_num = fragment_metadata_.size();
  auto dim_num = array_schema_->dim_num();

  // Get the number of fragments to process.
  unsigned num_fragments_to_process = 0;
  for (auto all_loaded : all_tiles_loaded_)
    num_fragments_to_process += !all_loaded;

  per_fragment_memory_ =
      memory_budget_ * memory_budget_ratio_coords_ / num_fragments_to_process;
  per_fragment_qc_memory_ = memory_budget_ *
                            memory_budget_ratio_query_condition_ /
                            num_fragments_to_process;

  // Create result tiles.
  if (subarray_.is_set()) {
    // Load as many tiles as the memory budget allows.
    auto status = parallel_for(
        storage_manager_->compute_tp(), 0, fragment_num, [&](uint64_t f) {
          auto range_it = result_tile_ranges_[f].rbegin();
          uint64_t t = 0;
          bool budget_exceeded = false;
          while (range_it != result_tile_ranges_[f].rend()) {
            for (t = range_it->first; t <= range_it->second; t++) {
              RETURN_NOT_OK(add_result_tile(
                  dim_num,
                  per_fragment_memory_,
                  per_fragment_qc_memory_,
                  f,
                  t,
                  fragment_metadata_[f]->array_schema(),
                  &budget_exceeded));
              *tiles_found = true;

              if (budget_exceeded)
                break;

              range_it->first++;
            }

            if (budget_exceeded) {
              logger_->debug(
                  "Budget exceeded adding result tiles, fragment {0}, tile {1}",
                  f,
                  t);

              if (result_tiles_[f].empty())
                return logger_->status(Status::SparseGlobalOrderReaderError(
                    "Cannot load a single tile for fragment, increase memory "
                    "budget"));
              break;
            }
            range_it++;
            remove_result_tile_range(f);
          }

          all_tiles_loaded_[f] = !budget_exceeded;
          return Status::Ok();
        });
    RETURN_NOT_OK_ELSE(status, logger_->status(status));
  } else {
    // Load as many tiles as the memory budget allows.
    auto status = parallel_for(
        storage_manager_->compute_tp(), 0, fragment_num, [&](uint64_t f) {
          uint64_t t = 0;
          auto tile_num = fragment_metadata_[f]->tile_num();
          bool budget_exceeded = false;

          // Figure out the start index.
          auto start = read_state_.frag_tile_idx_[f].first;
          if (!result_tiles_[f].empty()) {
            start = std::max(start, result_tiles_[f].back().tile_idx() + 1);
          }

          for (t = start; t < tile_num; t++) {
            RETURN_NOT_OK(add_result_tile(
                dim_num,
                per_fragment_memory_,
                per_fragment_qc_memory_,
                f,
                t,
                fragment_metadata_[f]->array_schema(),
                &budget_exceeded));
            *tiles_found = true;

            if (budget_exceeded) {
              logger_->debug(
                  "Budget exceeded adding result tiles, fragment {0}, tile {1}",
                  f,
                  t);

              if (result_tiles_[f].empty())
                return logger_->status(Status::SparseGlobalOrderReaderError(
                    "Cannot load a single tile for fragment, increase memory "
                    "budget"));
              break;
            }
          }

          all_tiles_loaded_[f] = !budget_exceeded;
          return Status::Ok();
        });
    RETURN_NOT_OK_ELSE(status, logger_->status(status));
  }

  bool done_adding_result_tiles = true;
  uint64_t num_rt = 0;
  for (unsigned int f = 0; f < fragment_num; f++) {
    num_rt += result_tiles_[f].size();
    done_adding_result_tiles &= all_tiles_loaded_[f] != 0;
  }

  logger_->debug("Done adding result tiles, num result tiles {0}", num_rt);

  if (done_adding_result_tiles) {
    logger_->debug("All result tiles loaded");
  }

  read_state_.done_adding_result_tiles_ = done_adding_result_tiles;
  return Status::Ok();
}

Status SparseGlobalOrderReader::compute_result_cell_slab(
    std::vector<ResultCellSlab>* result_cell_slabs) {
  auto timer_se = stats_->start_timer("compute_result_cell_slab");

  // First try to limit the maximum number of cells we copy using the size
  // of the output buffers for fixed sized attributes. Later we will validate
  // the memory budget. This is the first line of defence used to try to prevent
  // overflows when copying data.
  uint64_t num_cells = std::numeric_limits<uint64_t>::max();
  for (const auto& it : buffers_) {
    const auto& name = it.first;
    const auto size = *it.second.buffer_size_;
    if (array_schema_->var_size(name)) {
      auto temp_num_cells = size / constants::cell_var_offset_size;

      if (offsets_extra_element_ && temp_num_cells > 0)
        temp_num_cells--;

      num_cells = std::min(num_cells, temp_num_cells);
    } else {
      auto temp_num_cells = size / array_schema_->cell_size(name);
      num_cells = std::min(num_cells, temp_num_cells);
    }
  }

  // User gave us some empty buffers, exit.
  if (num_cells == 0) {
    zero_out_buffer_sizes();
    return Status::Ok();
  }

  if (array_schema_->cell_order() == Layout::HILBERT) {
    RETURN_CANCEL_OR_ERROR(merge_result_cell_slabs(
        num_cells,
        result_cell_slabs,
        HilbertCmpReverse(array_schema_->domain())));
  } else {
    RETURN_CANCEL_OR_ERROR(merge_result_cell_slabs(
        num_cells,
        result_cell_slabs,
        GlobalCmpReverse(array_schema_->domain())));
  }

  return Status::Ok();
}

template <class T>
Status SparseGlobalOrderReader::add_next_tile_to_queue(
    bool subarray_set,
    unsigned int frag_idx,
    uint64_t cell_idx,
    std::vector<std::list<ResultTileWithBitmap<uint8_t>>::iterator>&
        result_tiles_it,
    std::vector<uint8_t>& result_tile_used,
    std::priority_queue<ResultCoords, std::vector<ResultCoords>, T>& tile_queue,
    std::mutex& tile_queue_mutex,
    bool* need_more_tiles) {
  bool found = false;

  // Remove the tile from result tiles if it wasn't used at all.
  if (!result_tile_used[frag_idx]) {
    auto to_delete = result_tiles_it[frag_idx];
    to_delete--;
    remove_result_tile(frag_idx, to_delete);
  }

  // Try to find a tile.
  while (!found && result_tiles_it[frag_idx] != result_tiles_[frag_idx].end()) {
    found = !subarray_set;
    auto tile = &*result_tiles_it[frag_idx];

    // Find a cell that's in the subarray.
    if (subarray_set) {
      while (cell_idx < tile->cell_num()) {
        if (tile->bitmap_.size() == 0 || tile->bitmap_[cell_idx]) {
          found = true;
          break;
        }

        cell_idx++;
      }
    }

    // There was more tiles in this fragment, insert it in the queue.
    if (found) {
      std::unique_lock<std::mutex> ul(tile_queue_mutex);
      tile_queue.emplace(tile, cell_idx);
      result_tiles_it[frag_idx]++;
    } else {
      // Remove the tile.
      auto to_delete = result_tiles_it[frag_idx];
      result_tiles_it[frag_idx]++;
      remove_result_tile(frag_idx, to_delete);
    }

    result_tile_used[frag_idx] = false;

    // Once we move to the next tile, the saved cell index doesn't matter.
    cell_idx = 0;
  }

  if (!found) {
    // This fragment has more tiles potentially.
    if (!all_tiles_loaded_[frag_idx]) {
      // Set the next tile and return we need more tiles.
      read_state_.frag_tile_idx_[frag_idx].first++;
      read_state_.frag_tile_idx_[frag_idx].second = 0;
      *need_more_tiles = true;
    }
  }

  return Status::Ok();
}

Status SparseGlobalOrderReader::compute_hilbert_values(
    std::vector<ResultTile*>* result_tiles) {
  auto timer_se = stats_->start_timer("compute_hilbert_values");

  // For easy reference.
  auto dim_num = array_schema_->dim_num();

  // Create a Hilbet class.
  Hilbert h(dim_num);
  auto bits = h.bits();
  auto max_bucket_val = ((uint64_t)1 << bits) - 1;

  // Parallelize on tiles.
  auto status = parallel_for(
      storage_manager_->compute_tp(), 0, result_tiles->size(), [&](uint64_t t) {
        auto tile = (ResultTileWithBitmap<uint8_t>*)result_tiles->at(t);
        auto cell_num = tile->cell_num();
        auto rc = ResultCoords(tile, 0);
        std::vector<uint64_t> coords(dim_num);

        tile->hilbert_values_.resize(cell_num);
        for (rc.pos_ = 0; rc.pos_ < cell_num; rc.pos_++) {
          // Process only values in bitmap.
          if (tile->bitmap_.size() == 0 || tile->bitmap_[rc.pos_]) {
            // Compute Hilbert number for all dimensions first.
            for (uint32_t d = 0; d < dim_num; ++d) {
              auto dim = array_schema_->dimension(d);

              rc.pos_ = rc.pos_;
              coords[d] = dim->map_to_uint64(rc, d, bits, max_bucket_val);
            }

            // Now we are ready to get the final number.
            tile->hilbert_values_[rc.pos_] = h.coords_to_hilbert(&coords[0]);
          }
        }

        return Status::Ok();
      });
  RETURN_NOT_OK_ELSE(status, logger_->status(status));

  return Status::Ok();
}

template <class T>
Status SparseGlobalOrderReader::merge_result_cell_slabs(
    uint64_t num_cells, std::vector<ResultCellSlab>* result_cell_slabs, T cmp) {
  auto timer_se = stats_->start_timer("merge_result_cell_slabs");

  // TODO Parallelize.

  // For easy reference.
  bool subarray_set = subarray_.is_set();
  auto allows_dups = array_schema_->allows_dups();

  // A tile min heap, contains one Result coords per fragment.
  std::vector<ResultCoords> container;
  container.reserve(result_tiles_.size());
  std::priority_queue<ResultCoords, std::vector<ResultCoords>, T> tile_queue(
      cmp, std::move(container));

  std::mutex tile_queue_mutex;

  // If any fragments needs to load more tiles.
  bool need_more_tiles = false;

  // Tile iterators, per fragments.
  std::vector<std::list<ResultTileWithBitmap<uint8_t>>::iterator>
      result_tiles_it(result_tiles_.size());

  // Boolean per fragment that keeps track of when a tile is used.
  std::vector<uint8_t> result_tile_used(result_tiles_.size(), true);

  // For all fragments, get the first tile.
  auto status = parallel_for(
      storage_manager_->compute_tp(), 0, result_tiles_.size(), [&](uint64_t f) {
        if (result_tiles_[f].size() > 0) {
          // Initialize the iterator for this fragment.
          result_tiles_it[f] = result_tiles_[f].begin();

          // Get the cell index we were processing.
          auto cell_idx = read_state_.frag_tile_idx_[f].second;

          // Add the tile to the queue.
          RETURN_NOT_OK(add_next_tile_to_queue(
              subarray_set,
              f,
              cell_idx,
              result_tiles_it,
              result_tile_used,
              tile_queue,
              tile_queue_mutex,
              &need_more_tiles));
        }

        return Status::Ok();
      });
  RETURN_NOT_OK_ELSE(status, logger_->status(status));

  // Process all elements.
  while (!tile_queue.empty() && !need_more_tiles && num_cells > 0) {
    auto to_process = tile_queue.top();
    tile_queue.pop();

    // Process all cells with the same coordinates at once.
    while (!tile_queue.empty() && to_process.same_coords(tile_queue.top()) &&
           num_cells > 0) {
      // Potentially the next cell.
      auto next_tile = tile_queue.top();
      tile_queue.pop();

      // Take the cell with the highest fagment index.
      if (to_process.tile_->frag_idx() < next_tile.tile_->frag_idx()) {
        std::swap(to_process, next_tile);
      }

      // If we allow duplicates, create one slab for all the dups.
      if (allows_dups) {
        result_tile_used[next_tile.tile_->frag_idx()] = true;
        result_cell_slabs->emplace_back(next_tile.tile_, next_tile.pos_, 1);
        num_cells--;
        read_state_.frag_tile_idx_[next_tile.tile_->frag_idx()] =
            std::pair<uint64_t, uint64_t>(
                next_tile.tile_->tile_idx(), next_tile.pos_);
      }

      // Put the next cell in the queue.
      if (!next_tile.next()) {
        // Done with this tile, fetch another.
        RETURN_NOT_OK(add_next_tile_to_queue(
            subarray_set,
            next_tile.tile_->frag_idx(),
            0,
            result_tiles_it,
            result_tile_used,
            tile_queue,
            tile_queue_mutex,
            &need_more_tiles));
      } else {
        tile_queue.emplace(std::move(next_tile));
      }
    }

    if (num_cells == 0) {
      break;
    }

    result_tile_used[to_process.tile_->frag_idx()] = true;

    // Find how many cells to process using the top of the queue.
    auto& next_tile = tile_queue.top();

    // Temp result coord used to find the last position.
    ResultCoords temp_rc = to_process;

    // Check the top of the queue against last cell of the current tile.
    temp_rc.pos_ = to_process.tile_->cell_num() - 1;

    // If there is more than one fragment and we can't add the whole tile,
    // find the last possible cell in this tile smaller than the top of the
    // queue. Otherwise we are adding the whole tile.
    if (!tile_queue.empty() && cmp(temp_rc, next_tile)) {
      // Run a bisection seach on to find the last cell.
      uint64_t left = to_process.pos_;
      uint64_t right = temp_rc.pos_;
      while (left != right - 1) {
        // Check against mid.
        temp_rc.pos_ = left + (right - left) / 2;

        if (!cmp(temp_rc, next_tile))
          left = temp_rc.pos_;
        else
          right = temp_rc.pos_;
      }

      // Left is the last position smaller than the top of the queue.
      temp_rc.pos_ = left;
    }

    // Generate the result cell slabs.
    auto start = to_process.pos_;
    const auto& tile = to_process.tile_;
    const auto tile_idx = tile->tile_idx();
    auto& frag_idx = read_state_.frag_tile_idx_[tile->frag_idx()];

    // If no subarray is set, add all cells.
    auto tile_with_bitmap = (ResultTileWithBitmap<uint8_t>*)tile;
    if (!subarray_set || tile_with_bitmap->bitmap_.size() == 0) {
      auto length = std::min(temp_rc.pos_ - to_process.pos_ + 1, num_cells);
      result_cell_slabs->emplace_back(tile, start, length);
      frag_idx = std::pair<uint64_t, uint64_t>(tile_idx, start + length - 1);
      num_cells -= length;
    } else {
      // Process all cells, when there is a "hole" in the cell contiguity,
      // push a new cell slab.
      uint64_t length = 0;
      for (auto c = to_process.pos_; c <= temp_rc.pos_; c++) {
        if (!tile_with_bitmap->bitmap_[c]) {
          if (length != 0) {
            result_cell_slabs->emplace_back(tile, start, length);
            frag_idx =
                std::pair<uint64_t, uint64_t>(tile_idx, start + length - 1);
            num_cells -= length;
            length = 0;
          }

          start = c + 1;
        } else {
          length++;

          if (length == num_cells)
            break;
        }
      }

      // Add the last cell slab.
      if (length != 0) {
        result_cell_slabs->emplace_back(tile, start, length);
        frag_idx = std::pair<uint64_t, uint64_t>(tile_idx, start + length - 1);
        num_cells -= length;
      }
    }

    // Update the position in the tile.
    to_process.pos_ = temp_rc.pos_;

    // Put the next cell in the queue.
    if (!to_process.next()) {
      // Done with this tile, fetch another.
      RETURN_NOT_OK(add_next_tile_to_queue(
          subarray_set,
          tile->frag_idx(),
          0,
          result_tiles_it,
          result_tile_used,
          tile_queue,
          tile_queue_mutex,
          &need_more_tiles));
    } else {
      // Put the next cell on the queue to be resorted.
      read_state_.frag_tile_idx_[tile->frag_idx()] =
          std::pair<uint64_t, uint64_t>(tile->tile_idx(), to_process.pos_);
      tile_queue.emplace(std::move(to_process));
    }
  }

  logger_->debug(
      "Done merging result cell slabs, num slabs {0}",
      result_cell_slabs->size());

  return Status::Ok();
};

Status SparseGlobalOrderReader::remove_result_tile(
    const unsigned frag_idx,
    std::list<ResultTileWithBitmap<uint8_t>>::iterator rt) {
  // Remove coord tile size from memory budget.
  auto tile_idx = rt->tile_idx();
  uint64_t tiles_size = 0, tiles_size_qc = 0;
  RETURN_NOT_OK(get_coord_tiles_size<uint8_t>(
      true,
      array_schema_->dim_num(),
      frag_idx,
      tile_idx,
      &tiles_size,
      &tiles_size_qc));

  // Account for hilbert data.
  if (array_schema_->cell_order() == Layout::HILBERT) {
    tiles_size += fragment_metadata_[frag_idx]->cell_num(rt->tile_idx()) *
                  sizeof(uint64_t);
  }

  // Adjust per fragment memory usage.
  memory_used_for_coords_[frag_idx] -= tiles_size + sizeof(ResultTile);
  memory_used_for_qc_tiles_[frag_idx] -= tiles_size_qc;

  // Adjust total memory usage.
  {
    std::unique_lock<std::mutex> lck(mem_budget_mtx_);
    memory_used_for_coords_total_ -= tiles_size + sizeof(ResultTile);
    memory_used_qc_tiles_total_ -= tiles_size_qc;
  }

  // Delete the tile.
  result_tiles_[frag_idx].erase(rt);

  return Status::Ok();
}

Status SparseGlobalOrderReader::end_iteration() {
  // For easy reference.
  /*auto fragment_num = fragment_metadata_.size();

  // Remove the processed cell slabs.
  auto& new_front = read_state_.result_cell_slabs_[copy_end_.first - 1];

  // If the last cell slab processed wasn't processed fully, split it.
  if (new_front.length_ != copy_end_.second) {
    new_front.start_ += copy_end_.second;
    new_front.length_ -= copy_end_.second;
    copy_end_.first--;
  }

  // Clear result tiles that are not necessary anymore.
  auto cs_to_del_end = read_state_.result_cell_slabs_.begin() + copy_end_.first;

  // Last tile processed, initialized to the first tile in each fragments.
  std::vector<uint64_t> last_tile_processed(fragment_num);
  for (unsigned frag_idx = 0; frag_idx < fragment_num; frag_idx++) {
    if (!result_tiles_[frag_idx].empty())
      last_tile_processed[frag_idx] =
          result_tiles_[frag_idx].front().tile_idx();
  }

  // Record the last tile seen for each fragments in the slabs.
  for (auto it = read_state_.result_cell_slabs_.begin(); it < cs_to_del_end;
       it++) {
    last_tile_processed[it->tile_->frag_idx()] = it->tile_->tile_idx();
  }

  // Clear the tiles in each fragments until the front is the last seen.
  auto status = parallel_for(
      storage_manager_->compute_tp(), 0, fragment_num, [&](uint64_t f) {
        if (!result_tiles_[f].empty()) {
          while (result_tiles_[f].front().tile_idx() !=
                 last_tile_processed[f]) {
            RETURN_NOT_OK(remove_result_tile(f, result_tiles_[f].begin()));
          }
        }

        return Status::Ok();
      });

  auto uint64_t_max = std::numeric_limits<uint64_t>::max();
  copy_end_ = std::pair<uint64_t, uint64_t>(uint64_t_max, uint64_t_max);

  if (offsets_extra_element_) {
    RETURN_NOT_OK(add_extra_offset());
  }

  if (!incomplete()) {
    assert(memory_used_for_coords_total_ == 0);
    assert(memory_used_qc_tiles_total_ == 0);
    assert(memory_used_result_tile_ranges_ == 0);
  }

  uint64_t num_rt = 0;
  for (unsigned int f = 0; f < fragment_num; f++) {
    num_rt += result_tiles_[f].size();
  }
  empty_result_tiles_ = num_rt == 0;

  logger_->debug("Done with iteration, num result tiles {1}", num_rt);*/

  array_memory_tracker_->set_budget(std::numeric_limits<uint64_t>::max());
  return Status::Ok();
}

}  // namespace sm
}  // namespace tiledb
