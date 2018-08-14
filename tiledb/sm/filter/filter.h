/**
 * @file   filter.h
 *
 * @section LICENSE
 *
 * The MIT License
 *
 * @copyright Copyright (c) 2017-2018 TileDB, Inc.
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
 * This file declares class Filter.
 */

#ifndef TILEDB_FILTER_H
#define TILEDB_FILTER_H

#include "tiledb/sm/filter/filter_buffer.h"
#include "tiledb/sm/filter/filter_pipeline.h"
#include "tiledb/sm/misc/status.h"

namespace tiledb {
namespace sm {

/**
 * A Filter processes or modifies a byte region, modifying it in place, or
 * producing output in new buffer(s).
 *
 * Every filter implements both forward direction (used during write queries)
 * and a reverse direction (for reads).
 */
class Filter {
 public:
  /** Constructor. */
  Filter();

  /**
   * Returns a newly allocated clone of this Filter. The clone does not belong
   * to any pipeline. Caller is responsible for deletion of the clone.
   */
  Filter* clone() const;

  /**
   * Runs this filter in the "forward" direction (i.e. during write queries).
   *
   * Note: the input buffer should not be modified directly. The only reason it
   * is not a const pointer is to allow its offset field to be modified while
   * reading from it.
   *
   * Implemented by filter subclass.
   *
   * @param input Buffer with data to be filtered.
   * @param output Buffer with filtered data (unused by in-place filters).
   * @return
   */
  virtual Status run_forward(
      FilterBuffer* input, FilterBuffer* output) const = 0;

  /**
   * Runs this filter in the "reverse" direction (i.e. during read queries).
   *
   * Note: the input buffer should not be modified directly. The only reason it
   * is not a const pointer is to allow its offset field to be modified while
   * reading from it.
   *
   * Implemented by filter subclass.
   *
   * @param input Buffer with data to be filtered.
   * @param output Buffer with filtered data (unused by in-place filters).
   * @return
   */
  virtual Status run_reverse(
      FilterBuffer* input, FilterBuffer* output) const = 0;

  /** Sets the pipeline instance that executes this filter. */
  void set_pipeline(const FilterPipeline* pipeline);

 protected:
  /** Pointer to the pipeline instance that executes this filter. */
  const FilterPipeline* pipeline_;

  /**
   * Clone function must implemented by each specific Filter subclass. This is
   * used instead of copy constructors to allow for base-typed Filter instances
   * to be cloned without knowing their derived types.
   */
  virtual Filter* clone_impl() const = 0;
};

}  // namespace sm
}  // namespace tiledb

#endif  // TILEDB_FILTER_H
