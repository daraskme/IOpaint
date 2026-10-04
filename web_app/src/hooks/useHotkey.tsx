import { useStore } from "@/lib/states"
import { DependencyList } from "react"
import { useHotkeys, type HotkeyCallback } from "react-hotkeys-hook"

const useHotKey = (
  keys: string,
  callback: HotkeyCallback,
  deps?: DependencyList,
  // Match on the produced character (e.g. "[") instead of the physical key
  // code; needed for symbol keys and non-US layouts.
  useKey = false
) => {
  const disableShortCuts = useStore((state) => state.disableShortCuts)

  const ref = useHotkeys(keys, callback, { enabled: !disableShortCuts, useKey }, deps)
  return ref
}

export default useHotKey
