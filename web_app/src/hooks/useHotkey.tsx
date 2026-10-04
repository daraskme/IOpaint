import { useStore } from "@/lib/states"
import { DependencyList } from "react"
import { useHotkeys, type HotkeyCallback } from "react-hotkeys-hook"

const useHotKey = (
  keys: string,
  callback: HotkeyCallback,
  deps?: DependencyList
) => {
  const disableShortCuts = useStore((state) => state.disableShortCuts)

  const ref = useHotkeys(keys, callback, { enabled: !disableShortCuts }, deps)
  return ref
}

export default useHotKey
