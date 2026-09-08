import { afterEach } from "vitest"
import { cleanup } from "@testing-library/react"

// Each test mounts its own poster. Without this the previous one stays in the
// document and every getByText finds two.
afterEach(cleanup)
