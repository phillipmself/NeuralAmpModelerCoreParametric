#pragma once

#ifndef NDEBUG

#include <atomic>
#include <cassert>

namespace nam
{

/// \brief Debug-only reentrancy check for a parametric model's control/process API.
///
/// Asserts on construction that no other guarded call on the same flag is in flight, and
/// always clears it on destruction (including when unwinding through an exception), so a
/// single guard instance per call replaces a manual enter/leave pair plus a try/catch.
class DebugParamApiGuard
{
public:
  explicit DebugParamApiGuard(std::atomic_flag& flag) : _flag(flag)
  {
    assert(!_flag.test_and_set(std::memory_order_acquire));
  }
  ~DebugParamApiGuard() { _flag.clear(std::memory_order_release); }

  DebugParamApiGuard(const DebugParamApiGuard&) = delete;
  DebugParamApiGuard& operator=(const DebugParamApiGuard&) = delete;

private:
  std::atomic_flag& _flag;
};

} // namespace nam

#endif // NDEBUG
