/**
 * Public generation is deliberately unavailable until the project has a
 * reader-validated, reproducible method. Do not add a client-side fallback:
 * the server routes are also retired so static deployment cannot imply that
 * legacy output remains available.
 */
export default function App() {
  return (
    <main className="min-h-full bg-paper px-6 py-10 text-ink sm:px-10 sm:py-16">
      <section className="mx-auto max-w-2xl border-y border-haze py-10 sm:py-14">
        <p className="label mb-5">Palindrome research</p>
        <h1 className="font-display text-4xl font-bold tracking-[-0.045em] sm:text-6xl">
          Generation is unavailable.
        </h1>
        <p className="mt-8 max-w-xl text-lg leading-8 text-ash">
          This project is being rebuilt around exact validation and blinded
          human reader evidence. Earlier generated material has been withdrawn
          and is not presented as readable English.
        </p>
        <p className="mt-6 max-w-xl text-sm leading-6 text-ash">
          A new public result will appear only after it meets the project’s
          mechanical and reader-facing requirements.
        </p>
      </section>
    </main>
  )
}
