/* The composed poster, which is the site: v3 does not search, so the whole
 * composition arrives in one response. It is fed to the same pen in frames
 * anyway, and that is the thing worth testing — the writing is what shows a
 * reader that the two halves are built outward from one mirror. */
import { act, fireEvent, render, screen } from "@testing-library/react"
import { afterEach, beforeEach, describe, expect, test, vi } from "vitest"

// v3 is the bare path. The poster reads it once, when the module loads.
window.history.replaceState({}, "", "/")
const { default: App } = await import("./App")

const LEFT = ["one", "two", "three", "four"]
const RIGHT = ["five", "six", "seven", "eight"]

const shape = {
  center: "level", centerDisplay: "level", pivot: 2, pivotOdd: true,
  promptCenter: false,
  left: LEFT, right: RIGHT,
  letters: 30, words: 9,
}

/* The composition is synthesised into 14 frames 300ms apart, so the pen's span
   is the gap between them. One advance past the pen's ceiling finishes it. */
const FRAMES = 14
const GAP = 300
const MAX_SPAN = 1600

let composition: Record<string, unknown>
let health: Record<string, unknown> | null
let failure: { status: number; detail: string } | null

const answer = (body: unknown, ok = true, st = 200) =>
  Promise.resolve({ ok, status: st, json: async () => body })

const draw = async (ms: number) => {
  await act(async () => { await vi.advanceTimersByTimeAsync(ms) })
}
const drawAll = () => draw(FRAMES * GAP + MAX_SPAN)

const start = () => fireEvent.click(screen.getByRole("button", { name: /generate/i }))
const shown = (word: string) => screen.queryByText(word) !== null
const status = () => document.querySelector('label[for="p"]')?.textContent ?? ""

const mount = async () => {
  render(<App />)
  // The dial asks the bank for its ceiling on mount.
  await draw(0)
}

beforeEach(() => {
  vi.useFakeTimers()
  vi.advanceTimersByTime(1)
  health = { capacity: { novel: { max_letters: 9000 } } }
  failure = null
  composition = {
    shape, letters: 30, text: "One two three four level five six seven eight.",
    centre_is_yours: false,
  }
  vi.stubGlobal("fetch", vi.fn((url: string) => {
    if (url.startsWith("/api/v3/health")) return answer(health, health !== null)
    if (failure) return answer({ detail: failure.detail }, false, failure.status)
    return answer(composition)
  }))
})

afterEach(() => {
  vi.useRealTimers()
  vi.unstubAllGlobals()
})

describe("the composed poster", () => {
  test("takes the length ceiling from the bank", async () => {
    health = { capacity: { novel: { max_letters: 800 } } }
    await mount()

    // The dial cannot promise a length the material cannot reach.
    start()
    expect(status()).toBe("800 letters")
  })

  test("keeps its default length when the bank does not answer", async () => {
    health = null
    await mount()

    start()
    expect(status()).toBe("1,200 letters")
  })

  test("asks for the length on the dial, with the prompt as the centre", async () => {
    await mount()
    fireEvent.change(screen.getByPlaceholderText("add your own palindrome"), {
      target: { value: "never odd or even" },
    })
    start()

    const url = vi.mocked(fetch).mock.calls.at(-1)?.[0] as string
    expect(url).toContain("/api/v3/composition?letters=1200")
    expect(url).toContain("centre=never+odd+or+even")
  })

  test("writes the composition outward from the mirror", async () => {
    await mount()
    start()

    // Nothing is drawn by the response; the pen walks to it.
    expect(shown("four")).toBe(false)

    // Half the frames, and the pen settled on the last of them.
    await draw(GAP * (FRAMES / 2) + GAP - 10)
    expect([shown("four"), shown("three"), shown("two"), shown("one")])
      .toEqual([true, true, false, false])
    // The credits wait for the pen, not for the fetch.
    expect(screen.queryByRole("button", { name: /read it/i })).toBe(null)

    await drawAll()
    expect([shown("one"), shown("eight")]).toEqual([true, true])
    expect(status()).toBe("30 letters · 9 words")
  })

  test("says so when the centre is the visitor's own", async () => {
    composition = { ...composition, centre_is_yours: true }
    await mount()
    start()
    await drawAll()

    expect(status()).toBe("30 letters · 9 words · yours at the centre")
  })

  test("reads out the written form, not the bare letters", async () => {
    await mount()
    start()
    await drawAll()

    fireEvent.click(screen.getByRole("button", { name: /read it/i }))
    expect(shown("One two three four level five six seven eight.")).toBe(true)
  })

  test("reports what the composer said when it refuses", async () => {
    failure = { status: 422, detail: "no centre matches that" }
    await mount()
    start()
    await draw(0)

    expect(status()).toBe("no centre matches that")
    expect(shown("four")).toBe(false)
  })
})
