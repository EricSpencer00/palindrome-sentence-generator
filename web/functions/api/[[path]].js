// The "worker" half of the site: Pages serves the static page, and every /api
// call is proxied to the configured API origin. Same-origin
// for the browser, so no CORS, and the backend hostname stays an implementation
// detail. The upstream response is returned as-is so the SSE body keeps
// streaming rather than being buffered.
const ORIGIN = "https://palindrome-api.ericspencer.us"

// The upstream service still contains historical generators.  Keep the
// public API honest at the Pages boundary: health and other diagnostic
// endpoints remain available, but no generation route may expose material
// before an independently generated candidate has passed the reader gate.
const RETIRED_OUTPUT = /^\/api\/(?:generate|v\d+\/(?:generate|paragraph|composition|palindrome|refrain))(?:\/|$)/
const RETIREMENT = {
  detail:
    "Palindrome output is retired: exactness and programmatic filters do not establish readable English. The service will remain unavailable until independently generated candidates have blinded human-reader evidence.",
}

export async function onRequest(context) {
  const url = new URL(context.request.url)

  if (RETIRED_OUTPUT.test(url.pathname)) {
    return new Response(JSON.stringify(RETIREMENT), {
      status: 503,
      headers: { "Content-Type": "application/json", "Cache-Control": "no-store" },
    })
  }

  const upstream = new URL(url.pathname + url.search, ORIGIN)

  const res = await fetch(upstream, {
    method: context.request.method,
    headers: context.request.headers,
    body: ["GET", "HEAD"].includes(context.request.method) ? undefined : context.request.body,
  })

  const headers = new Headers(res.headers)
  headers.set("Cache-Control", "no-cache")
  headers.delete("content-encoding")   // never re-encode a stream we pass through
  headers.delete("content-length")
  return new Response(res.body, { status: res.status, headers })
}
