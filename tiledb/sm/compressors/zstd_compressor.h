/**
 * @file   zstd_compressor.h
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
 * This file defines the zstd compressor class.
 */

#ifndef TILEDB_ZSTD_H
#define TILEDB_ZSTD_H

#include "tiledb.h"
#include "tiledb/common/status.h"

using namespace tiledb::common;

namespace tiledb {
namespace sm {

class Buffer;
class ConstBuffer;
class PreallocatedBuffer;

/** Handles compression/decompression with the zstd library. */
class ZStd {
 public:
  /**
   * Compression function.
   *
   * @param level Compression level.
   *  Use TILEDB_FILTER_ZSTD_DEFAULT_LEVEL for ZStd's default level.
   * @param input_buffer Input buffer to read from.
   * @param output_buffer Output buffer to write to the compressed data.
   * @return Status
   */
  static Status compress(
      int level, ConstBuffer* input_buffer, Buffer* output_buffer);

  /**
   * Overloaded compression function with default compression level.
   *
   * @param input_buffer Input buffer to read from.
   * @param output_buffer Output buffer to write to the compressed data.
   * @return Status
   */
  static Status compress(ConstBuffer* input_buffer, Buffer* output_buffer);

  /**
   * Decompression function.
   *
   * @param input_buffer Input buffer to read from.
   * @param output_buffer Output buffer to write the decompressed data to.
   * @return Status
   */
  static Status decompress(
      ConstBuffer* input_buffer, PreallocatedBuffer* output_buffer);

  /** Returns the default compression level. */
  static int default_level() {
    return TILEDB_FILTER_ZSTD_DEFAULT_LEVEL;
  }

  /** Returns the compression overhead for the given input. */
  static uint64_t overhead(uint64_t nbytes);
};

}  // namespace sm
}  // namespace tiledb

#endif  // TILEDB_ZSTD_H
