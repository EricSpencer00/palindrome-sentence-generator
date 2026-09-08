/* The streaming poster: what it does with the frames, and when it draws them.
 *
 * Two things are worth pinning down. The stream contract — status, plan,
 * partial, result, error — because the poster is its only reader, and a change
 * on either side breaks nothing loudly. And the pen, because `done` is set when
 * the pen arrives rather than when the result lands: every version of that rule
 * that got it the other way round shipped a page which said "searching" for
 * ever. */
import { act, fireEvent, render, screen } from "@testing-library/react"
import { afterEach, beforeEach, describe, expect, test, vi } from "vitest"

/* jsdom opens at "/", which is v3 — composed, not searched, and it does not
 * stream. The poster reads the path once when the module loads, so the path has
 * to be set before the import. */
window.history.replaceState({}, "", "/v1")
const { default: App } = await import("./App")

/* EventSource does not exist in jsdom, and a real one would need a server. This
 * is the whole surface the poster uses: two handlers and close(). */
class FakeStream {
  static last: FakeStream | null = null
  onmessage: ((e: { data: string }) => void) | null = null
  onerror: (() => void) | null = null
  closed = false
  readonly url: string

  constructor(url: string) {
    this.url = url
    FakeStream.last = this
  }

  close() { this.closed = true }

  send(msg: unknown) { act(() => { this.onmessage?.({ data: JSON.stringify(msg) }) }) }

  fail() { act(() => { this.onerror?.() }) }
}

const stream = () => {
  const s = FakeStream.last
  if (!s) throw new Error("the poster never opened a stream")
  return s
}

/* A draft. The words are all different, so a query for one of them cannot match
 * its own mirror image — which on a real palindrome it always would. */
const draft = (left: string[], right: string[]) => ({
  type: "partial",
  center: "level", centerDisplay: "level", pivot: 2, pivotOdd: true,
  promptCenter: false,
  left, right,
  letters: [...left, "level", ...right].join("").length,
  words: left.length + right.length + 1,
})

const answer = (left: string[], right: string[]) => ({
  ...draft(left, right), type: "result", lm: null, coherence: 0.5, seconds: 2,
})

const LEFT = ["one", "two", "three", "four"]
const RIGHT = ["five", "six", "seven", "eight"]

/* The pen's own numbers. It steps every 16ms across a span taken from the gap
 * between the last two frames, clamped to [200, 1600]. Frames sent back to back
 * therefore draw over 200ms, and one advance past the ceiling finishes the pen
 * whatever the gap turned out to be. */
const STEP = 16
const MIN_SPAN = 200
const MAX_SPAN = 1600

const draw = async (ms: number) => {
  await act(async () => { await vi.advanceTimersByTimeAsync(ms) })
}
const drawHalf = () => draw(MIN_SPAN / 2 + STEP)
const drawAll = () => draw(MAX_SPAN + STEP)

const start = () => fireEvent.click(screen.getByRole("button", { name: /generate/i }))
const shown = (word: string) => screen.queryByText(word) !== null
const status = () => document.querySelector('label[for="p"]')?.textContent ?? ""

beforeEach(() => {
  vi.useFakeTimers()
  /* The fake clock starts at zero, and the pen reads a zero timestamp as "no
     previous frame" and falls back to its default span. A real page cannot open
     a stream in its first millisecond, so the tests should not either. */
  vi.advanceTimersByTime(1)
  FakeStream.last = null
  vi.stubGlobal("EventSource", FakeStream)
  render(<App />)
})

afterEach(() => {
  vi.useRealTimers()
  vi.unstubAllGlobals()
})

describe("the stream", () => {
  test("opens the v1 endpoint with the prompt", () => {
    fireEvent.change(screen.getByPlaceholderText("never odd or even"), {
      target: { value: "never odd or even" },
    })
    start()
    expect(stream().url).toBe("/api/generate?prompt=never%20odd%20or%20even&budget=16")
  })

  test("reports elapsed time from status frames and draws nothing", () => {
    start()
    stream().send({ type: "status", elapsed: 1.25 })

    expect(status()).toBe("1.3s")
    expect(shown("level")).toBe(false)
  })

  test("draws a draft outward from the mirror, a word at a time", async () => {
    start()
    stream().send({ type: "plan", expectLetters: 200 })
    stream().send(draft(LEFT, RIGHT))

    // Nothing is written by the frame itself; the pen walks to it.
    expect(shown("four")).toBe(false)

    await drawHalf()
    // Two of the four on each side, and they are the two nearest the mirror.
    expect([shown("four"), shown("three"), shown("two"), shown("one")])
      .toEqual([true, true, false, false])
    expect([shown("five"), shown("six"), shown("seven"), shown("eight")])
      .toEqual([true, true, false, false])

    await drawAll()
    expect([shown("one"), shown("eight")]).toEqual([true, true])
  })

  test("leaves drawn words in their cells when the draft grows", async () => {
    start()
    /* The grid is picked once, from the size the search says it is aiming at.
       Picking it again from the draft in hand would rewrap, and a rewrap moves
       words that are already written. */
    stream().send({ type: "plan", expectLetters: 2000 })
    stream().send(draft(LEFT, RIGHT))
    await drawAll()

    const cell = () => {
      const el = screen.getByText("four").parentElement
      return `${el?.style.left} ${el?.style.top}`
    }
    const before = cell()

    /* A beam search grows the draft by twenty-odd words at a time. It prepends
       on the left and appends on the right, so every word already on screen
       keeps the cell it landed in — however much longer the draft gets. */
    const grown = (tag: string) => Array.from({ length: 30 }, (_, i) => tag + i)
    stream().send(draft([...grown("l"), ...LEFT], [...RIGHT, ...grown("r")]))
    await drawAll()

    expect(cell()).toBe(before)
    expect([shown("l0"), shown("r0")]).toEqual([true, true])
  })

  test("closes the stream on the result and waits for the pen", async () => {
    start()
    stream().send(answer(LEFT, RIGHT))

    expect(stream().closed).toBe(true)
    // The result has landed and the credits are still not up: the pen decides.
    expect(screen.queryByRole("button", { name: /read it/i })).toBe(null)

    await drawAll()
    expect(screen.queryByRole("button", { name: /read it/i })).not.toBe(null)
    expect(status()).toBe("37 letters · 9 words · 2s")
  })

  test("reads out the whole palindrome once the run is done", async () => {
    start()
    stream().send(answer(LEFT, RIGHT))
    await drawAll()

    fireEvent.click(screen.getByRole("button", { name: /read it/i }))
    expect(screen.getByText(/one two three four/).textContent)
      .toContain("five six seven eight")
  })

  test("reports an error frame and stops", () => {
    start()
    stream().send({ type: "error", message: "the bank is empty" })

    expect(status()).toBe("the bank is empty")
    expect(stream().closed).toBe(true)
    const button = screen.getByRole("button", { name: /generate/i }) as HTMLButtonElement
    expect(button.disabled).toBe(false)
  })

  test("reports a transport failure while searching", () => {
    start()
    stream().fail()

    expect(status()).toBe("could not reach the generator")
    expect(stream().closed).toBe(true)
  })

  test("ignores a transport failure once the run is done", async () => {
    start()
    stream().send(answer(LEFT, RIGHT))
    await drawAll()

    // A server closing a finished stream is normal, and must not paint an error
    // over a poster that is already written.
    stream().fail()
    expect(status()).toBe("37 letters · 9 words · 2s")
  })

  test("clears the poster when the next run starts", async () => {
    start()
    const first = stream()
    first.send(answer(LEFT, RIGHT))
    await drawAll()

    start()
    expect(stream()).not.toBe(first)
    // Nothing of the finished poster survives into the new run.
    expect(shown("four")).toBe(false)
    expect(status()).toBe("0.0s")
  })
})
