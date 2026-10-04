import { useStore } from "@/lib/states"
import { Button } from "./ui/button"

const InteractiveSegConfirmActions = () => {
  const [
    interactiveSegState,
    resetInteractiveSegState,
    handleInteractiveSegAccept,
  ] = useStore((state) => [
    state.interactiveSegState,
    state.resetInteractiveSegState,
    state.handleInteractiveSegAccept,
  ])

  if (!interactiveSegState.isInteractiveSeg) {
    return null
  }

  return (
    <div className="z-10 absolute top-[68px] rounded-xl border-solid border p-[8px] left-1/2 translate-x-[-50%] flex justify-center items-center gap-[8px] bg-background">
      <Button
        onClick={() => {
          resetInteractiveSegState()
        }}
        size="sm"
        variant="secondary"
      >
        Cancel
      </Button>
      <Button
        size="sm"
        onClick={() => {
          handleInteractiveSegAccept()
        }}
      >
        Accept
      </Button>
    </div>
  )
}

interface ItemProps {
  x: number
  y: number
  positive: boolean
}

const Item = (props: ItemProps) => {
  const { x, y, positive } = props
  const name = positive
    ? "bg-[rgba(21,_215,_121,_0.936)] outline-[rgba(98,255,179,0.31)]"
    : "bg-[rgba(237,_49,_55,_0.942)] outline-[rgba(255,89,95,0.31)]"
  return (
    <div
      className={`absolute h-[10px] w-[10px] rounded-[50%] ${name} outline-8 outline`}
      style={{
        left: x,
        top: y,
        transform: "translate(-50%, -50%)",
      }}
    />
  )
}

const InteractiveSegPoints = () => {
  const clicks = useStore((state) => state.interactiveSegState.clicks)

  return (
    <div className="absolute h-full w-full overflow-hidden pointer-events-none">
      {clicks.map((click) => {
        return (
          <Item
            key={click[3]}
            x={click[0]}
            y={click[1]}
            positive={click[2] === 1}
          />
        )
      })}
    </div>
  )
}

const InteractiveSeg = () => {
  return (
    <div>
      <InteractiveSegConfirmActions />
      {/* <InteractiveSegReplaceModal /> */}
    </div>
  )
}

export { InteractiveSeg, InteractiveSegPoints }
