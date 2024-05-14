/* Copyright 2020 The TensorFlow Authors. All Rights Reserved.

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
#include "tsl/profiler/utils/per_thread.h"

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "absl/cleanup/cleanup.h"
#include "absl/synchronization/blocking_counter.h"
#include "absl/synchronization/notification.h"
#include "tsl/platform/env.h"
#include "tsl/platform/test.h"

namespace tsl {
namespace profiler {
namespace {

enum ProfilingStage {
  kBeforeProfiling = 1,
  kDuringProfiling = 2,
  kAfterProfiling = 3,
  kNever = 4
};

struct ThreadSyncControl {
  ThreadSyncControl()
      : could_start_profiling(4), could_stop_profiling(6), could_exit_all(6) {}

  absl::Notification profiling_started;
  absl::Notification profiling_stopped;
  absl::Notification exiting_all;

  absl::BlockingCounter could_start_profiling;
  absl::BlockingCounter could_stop_profiling;
  absl::BlockingCounter could_exit_all;
};

static ThreadSyncControl& GetSyncContols() {
  static ThreadSyncControl* control = new ThreadSyncControl();
  return *control;
}

void thread_main(ProfilingStage firstUseStage, ProfilingStage exitStage,
                 int32_t id) {
  if (firstUseStage == kBeforeProfiling) {
    auto& td = PerThread<int32_t>::Get();
    td = id;
    GetSyncContols().could_start_profiling.DecrementCount();
  }
  if (exitStage == kBeforeProfiling) {
    return;
  }
  GetSyncContols().profiling_started.WaitForNotification();

  if (firstUseStage == kDuringProfiling) {
    auto& td = PerThread<int32_t>::Get();
    td = id;
    GetSyncContols().could_stop_profiling.DecrementCount();
  }
  if (exitStage == kDuringProfiling) {
    return;
  }
  GetSyncContols().profiling_stopped.WaitForNotification();

  if (firstUseStage == kAfterProfiling) {
    auto& td = PerThread<int32_t>::Get();
    td = id;
    GetSyncContols().could_exit_all.DecrementCount();
  }
  if (exitStage == kAfterProfiling) {
    return;
  }
  GetSyncContols().exiting_all.WaitForNotification();
}

#define StartThread(id, firstUseStage, exitStage)            \
  auto* thread##id = Env::Default()->StartThread(            \
      ThreadOptions(), "thread_##id",                        \
      [=]() { thread_main(firstUseStage, exitStage, id); }); \
  auto cleanup_thread##id =                                  \
      absl::MakeCleanup([&thread##id] { delete thread##id; });

using ::testing::ElementsAre;
using ::testing::WhenSorted;

TEST(PerThreadRecordingTest, Lifecycles) {
  auto get_ids = [](std::vector<std::shared_ptr<int32_t>>& threads_data) {
    std::vector<int> threads_values;
    for (const auto& ptd : threads_data) {
      threads_values.push_back(*ptd);
    }
    return threads_values;
  };

  auto threads_data = PerThread<int32_t>::StartRecording();
  auto threads_values = get_ids(threads_data);
  EXPECT_THAT(threads_values, ::testing::SizeIs(0));

  StartThread(111, kBeforeProfiling, kBeforeProfiling);
  StartThread(112, kBeforeProfiling, kDuringProfiling);
  StartThread(113, kBeforeProfiling, kAfterProfiling);
  StartThread(114, kBeforeProfiling, kNever);

  StartThread(122, kDuringProfiling, kDuringProfiling);
  StartThread(123, kDuringProfiling, kAfterProfiling);
  StartThread(124, kDuringProfiling, kNever);

  StartThread(133, kAfterProfiling, kAfterProfiling);
  StartThread(134, kAfterProfiling, kNever);

  // These thread will never initialize the Per Thread data
  StartThread(141, kNever, kBeforeProfiling);
  StartThread(142, kNever, kDuringProfiling);
  StartThread(143, kNever, kAfterProfiling);
  StartThread(144, kNever, kNever);

  GetSyncContols().could_start_profiling.Wait();

  threads_data = PerThread<int32_t>::StopRecording();
  threads_values = get_ids(threads_data);
  EXPECT_THAT(threads_values, WhenSorted(ElementsAre(111, 112, 113, 114)));

  // Start again, thread 111 already exit
  threads_data = PerThread<int32_t>::StartRecording();
  threads_values = get_ids(threads_data);
  EXPECT_THAT(threads_values, WhenSorted(ElementsAre(112, 113, 114)));

  GetSyncContols().profiling_started.Notify();

  StartThread(222, kDuringProfiling, kDuringProfiling);
  StartThread(223, kDuringProfiling, kAfterProfiling);
  StartThread(224, kDuringProfiling, kNever);

  StartThread(233, kAfterProfiling, kAfterProfiling);
  StartThread(234, kAfterProfiling, kNever);

  StartThread(242, kNever, kDuringProfiling);
  StartThread(243, kNever, kAfterProfiling);
  StartThread(244, kNever, kNever);

  GetSyncContols().could_stop_profiling.Wait();

  threads_data = PerThread<int32_t>::StopRecording();
  threads_values = get_ids(threads_data);
  EXPECT_THAT(threads_values, WhenSorted(ElementsAre(112, 113, 114, 122, 123,
                                                     124, 222, 223, 224)));

  threads_data = PerThread<int32_t>::StartRecording();
  threads_values = get_ids(threads_data);
  EXPECT_THAT(threads_values,
              WhenSorted(ElementsAre(113, 114, 123, 124, 223, 224)));

  GetSyncContols().profiling_stopped.Notify();

  StartThread(333, kAfterProfiling, kAfterProfiling);
  StartThread(334, kAfterProfiling, kNever);

  StartThread(343, kNever, kAfterProfiling);
  StartThread(344, kNever, kNever);

  GetSyncContols().could_exit_all.Wait();

  threads_data = PerThread<int32_t>::StopRecording();
  threads_values = get_ids(threads_data);
  EXPECT_THAT(threads_values,
              WhenSorted(ElementsAre(113, 114, 123, 124, 133, 134, 223, 224,
                                     233, 234, 333, 334)));

  GetSyncContols().exiting_all.Notify();
}

}  // namespace
}  // namespace profiler
}  // namespace tsl
