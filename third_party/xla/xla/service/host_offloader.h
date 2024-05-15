/* Copyright 2024 The OpenXLA Authors.

 Licensed under the Apache License, Version 2.0 (the "License");
 you may not use this file except in compliance with the License.
 You may obtain a copy of the License at

     http://www.apache.org/licenses/LICENSE-2.0

 Unless required by applicable law or agreed to in writing, software
 distributed under the License is distributed on an "AS IS" BASIS,
 WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 See the License for the specific language governing permissions and
 limitations under the License.
 ==============================================================================*/
#ifndef XLA_SERVICE_HOST_OFFLOADER_H_
#define XLA_SERVICE_HOST_OFFLOADER_H_

#include <cstdint>
#include <memory>
#include <string>

#include "absl/container/flat_hash_set.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/service/hlo_alias_analysis.h"
#include "xla/service/hlo_buffer.h"
#include "xla/service/hlo_pass_interface.h"

namespace xla {

class HloCostAnalysis;

struct InstructionAndShapeIndex {
  InstructionAndShapeIndex(HloInstruction* instruction)
      : instruction(instruction) {}
  InstructionAndShapeIndex(HloInstruction* instruction, ShapeIndex shape_index)
      : instruction(instruction), shape_index(shape_index) {}
  HloInstruction* instruction;
  ShapeIndex shape_index;
  std::string ToString() const;

  template <typename H>
  static H Hash(H h, const InstructionAndShapeIndex& i) {
    h = H::combine(std::move(h), i.instruction);
    h = H::combine(std::move(h), i.shape_index);
    return std::move(h);
  }

  template <typename H>
  friend H AbslHashValue(H h, const InstructionAndShapeIndex& i) {
    return InstructionAndShapeIndex::Hash(std::move(h), i);
  }
};

bool operator==(const InstructionAndShapeIndex& lhs,
                const InstructionAndShapeIndex& rhs);

// This pass does "host memory offloading". If a tensor is annotated to be moved
// to or from the host, this pass will remove the annotations and update each
// tensor's layout with host memory spaces and insert copies if necessary. This
// pass checks to make sure that no compute is done on the tensors annotated for
// host memory offload; if there is compute, it is considered a user error and
// an error will be returned.
// The pass will "walk down" the Hlo graph starting from either MoveToHost
// custom calls or from parameters with host memory space in their layout. All
// tensors along each path have their memory space set as host memory space. If
// a MoveToHost custom call is paired with a DynamicUpdateSlice, the
// DynamicUpdateSlice will write into host memory space. Otherwise, a copy from
// device to host will be inserted. All MoveToHost and MoveToDevice custom calls
// are removed by the end of this pass.
class HostOffloader : public HloModulePass {
 public:
  explicit HostOffloader(int64_t host_memory_space_color)
      : kHostMemorySpaceColor(host_memory_space_color) {}
  ~HostOffloader() override = default;

  absl::string_view name() const override { return "host-offloader"; }

  using HloPassInterface::Run;
  absl::StatusOr<bool> Run(
      HloModule* module,
      const absl::flat_hash_set<absl::string_view>& execution_threads) override;

 private:
  const int64_t kHostMemorySpaceColor;
  absl::flat_hash_set<HloInstruction*>
      already_visited_move_to_host_custom_calls_;
  absl::flat_hash_set<HloInstruction*> dynamic_update_slices_already_handled_;
  absl::flat_hash_map<HloInstruction*, HloInstruction*> copies_created_after_;
  absl::flat_hash_set<InstructionAndShapeIndex> already_inserted_copy_before_;

  absl::Status DynamifySlice(HloInstruction* slice);
  bool IsValidDuringPureMemoryOffload(const HloInstruction* instruction) const;
  bool InstructionIsAllowedBetweenMoveToHostAndDus(
      const HloInstruction* instruction) const;
  bool InstructionIsAllowedBetweenDsAndMoveToDevice(
      const HloInstruction* instruction) const;
  absl::StatusOr<bool> HandleInputStreaming(HloComputation* entry_computation);
  absl::StatusOr<bool> HandleMoveToHostCustomCall(
      HloInstruction* custom_call_instruction);
  absl::StatusOr<bool> HandleMoveToDeviceCustomCall(
      HloInstruction* custom_call_instruction);
  absl::Status CreateAllocateBufferForDynamicUpdateSlice(
      HloInstruction* dynamic_update_slice);
  absl::Status ValidateSliceLeadsToMoveToDeviceCustomCall(
      HloInstruction* instruction) const;
  absl::StatusOr<bool> WalkDownHostMemoryOffloadPaths(
      const InstructionAndShapeIndex& starting_instruction_and_index,
      bool insert_copy_before);
  std::vector<InstructionAndShapeIndex> GetStartingInstructions(
      HloInstruction* custom_call_instruction);
  absl::StatusOr<bool> InsertCopyBetween(
      const InstructionAndShapeIndex& before_instruction_and_index,
      const InstructionAndShapeIndex& after_instruction_and_index);
  absl::StatusOr<bool> ApplySchedulingFix(
      HloModule* module,
      const absl::flat_hash_set<absl::string_view>& execution_threads);
};

}  // namespace xla

#endif  // XLA_SERVICE_HOST_OFFLOADER_H_
