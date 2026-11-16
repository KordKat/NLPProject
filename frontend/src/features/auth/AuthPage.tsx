import { useState, type FormEvent } from "react";
import { Atom, BookOpenCheck, Network, Sigma } from "lucide-react";
import { Logo, Spinner } from "../../components/ui";
import { useAuth } from "../../hooks/useAuth";

export default function AuthPage() {
  const { login, register } = useAuth();
  const [mode, setMode] = useState<"login" | "register">("login");
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [name, setName] = useState("");
  const [error, setError] = useState("");
  const [busy, setBusy] = useState(false);

  async function submit(e: FormEvent) {
    e.preventDefault();
    setError("");
    setBusy(true);
    try {
      if (mode === "login") await login(email, password);
      else await register(email, password, name || email.split("@")[0]);
    } catch (err: any) {
      setError(err?.message || "Something went wrong");
    } finally {
      setBusy(false);
    }
  }

  return (
    <div className="grid min-h-full lg:grid-cols-[1.1fr_1fr]">
      <section className="relative hidden flex-col justify-between overflow-hidden border-r border-white/5 p-12 lg:flex">
        <Logo />
        <div className="max-w-lg">
          <h1 className="text-4xl font-semibold leading-tight tracking-tight text-slate-50">
            A scientific operating system for <span className="bg-gradient-to-r from-accent-sky to-accent-mint bg-clip-text text-transparent">your research</span>.
          </h1>
          <p className="mt-4 text-slate-400">
            Organise work in Projects. KeawGPT retrieves arXiv papers, explores a growing knowledge graph, searches the web, computes, plots and writes cited reports — and remembers your project context.
          </p>
          <ul className="mt-8 grid grid-cols-2 gap-3 text-sm text-slate-300">
            {[
              [BookOpenCheck, "Evidence-grounded answers with citations"],
              [Network, "Neo4j knowledge graph that grows with you"],
              [Sigma, "Math, Python and visualizations"],
              [Atom, "Reports in Markdown and PDF"],
            ].map(([Icon, label], i) => {
              const I = Icon as typeof Atom;
              return (
                <li key={i} className="panel flex items-start gap-2 p-3">
                  <I size={16} className="mt-0.5 shrink-0 text-accent-teal" />
                  <span>{label as string}</span>
                </li>
              );
            })}
          </ul>
        </div>
        <p className="text-xs text-slate-500">Powered by KeawGPT · FAISS · Neo4j</p>
      </section>

      <section className="flex items-center justify-center p-6">
        <form onSubmit={submit} className="panel w-full max-w-sm p-7" aria-label={mode === "login" ? "Sign in" : "Create account"}>
          <div className="mb-6 lg:hidden">
            <Logo />
          </div>
          <h2 className="text-xl font-semibold text-slate-50">{mode === "login" ? "Welcome back" : "Create your account"}</h2>
          <p className="mb-6 mt-1 text-sm text-slate-400">{mode === "login" ? "Sign in to continue your research." : "Start building your research workspace."}</p>
          {mode === "register" && (
            <div className="mb-3">
              <label className="label" htmlFor="name">Display name</label>
              <input id="name" className="input" value={name} onChange={(e) => setName(e.target.value)} autoComplete="name" />
            </div>
          )}
          <div className="mb-3">
            <label className="label" htmlFor="email">Email</label>
            <input id="email" type="email" required className="input" value={email} onChange={(e) => setEmail(e.target.value)} autoComplete="email" />
          </div>
          <div className="mb-4">
            <label className="label" htmlFor="password">Password</label>
            <input
              id="password"
              type="password"
              required
              minLength={mode === "register" ? 8 : 1}
              className="input"
              value={password}
              onChange={(e) => setPassword(e.target.value)}
              autoComplete={mode === "login" ? "current-password" : "new-password"}
            />
            {mode === "register" && <p className="mt-1 text-xs text-slate-500">At least 8 characters.</p>}
          </div>
          {error && <p className="mb-3 rounded-lg border border-rose-500/30 bg-rose-500/10 px-3 py-2 text-sm text-rose-200" role="alert">{error}</p>}
          <button className="btn-primary w-full" disabled={busy}>
            {busy && <Spinner />} {mode === "login" ? "Sign in" : "Create account"}
          </button>
          <p className="mt-5 text-center text-sm text-slate-400">
            {mode === "login" ? "New to STEM OS?" : "Already have an account?"}{" "}
            <button type="button" className="font-medium text-accent-sky hover:underline" onClick={() => { setMode(mode === "login" ? "register" : "login"); setError(""); }}>
              {mode === "login" ? "Create an account" : "Sign in"}
            </button>
          </p>
        </form>
      </section>
    </div>
  );
}
