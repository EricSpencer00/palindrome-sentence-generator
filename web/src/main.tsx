import { StrictMode } from "react"
import { createRoot } from "react-dom/client"
import "./index.css"
import App from "./App.tsx"
import DevV4 from "./DevV4.tsx"

/* Three routes, one bundle.
 *
 *   /      the poster — v1 writes a palindrome and you watch it happen
 *   /dev   the reader — v4 composes one from verified material and punctuates it
 *   /v2    the poster again, against v2, which places whole attested sentences
 *
 * `App` covers / and /v2 because they are the same page pointed at different
 * endpoints. /dev is a different page: v4 does not stream, so there is nothing
 * to animate and the text is the whole of it. See `public/_redirects`, without
 * which Pages 404s both paths before any of this runs. */
const path = window.location.pathname.replace(/\/+$/, "")
const Page = path === "/dev" ? DevV4 : App

createRoot(document.getElementById("root")!).render(
  <StrictMode>
    <Page />
  </StrictMode>,
)
