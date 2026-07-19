import { useEffect, useState } from 'react'
import './App.css'

const API = import.meta.env.VITE_API_URL || 'http://localhost:8080'

export default function App() {
  const [token, setToken] = useState(localStorage.getItem('tt_token') || '')
  const [user, setUser] = useState('admin')
  const [pass, setPass] = useState('admin')
  const [err, setErr] = useState('')
  const [status, setStatus] = useState(null)
  const [trades, setTrades] = useState({ open: [], closed: [] })

  async function api(path, opt = {}) {
    const headers = { 'Content-Type': 'application/json', ...(opt.headers || {}) }
    if (token) headers.Authorization = `Bearer ${token}`
    const res = await fetch(API + path, { ...opt, headers })
    if (!res.ok) {
      const body = await res.json().catch(() => ({}))
      throw new Error(body.detail || res.statusText)
    }
    return res.json()
  }

  async function login() {
    setErr('')
    try {
      const d = await api('/api/login', {
        method: 'POST',
        body: JSON.stringify({ username: user, password: pass }),
      })
      localStorage.setItem('tt_token', d.token)
      setToken(d.token)
    } catch (e) {
      setErr(String(e.message || e))
    }
  }

  async function refresh() {
    if (!token) return
    try {
      const [s, t] = await Promise.all([api('/api/status'), api('/api/trades')])
      setStatus(s)
      setTrades(t)
    } catch (e) {
      setErr(String(e.message || e))
      if (String(e.message).includes('401') || String(e.message).toLowerCase().includes('auth')) {
        setToken('')
        localStorage.removeItem('tt_token')
      }
    }
  }

  useEffect(() => {
    refresh()
    const id = setInterval(refresh, 15000)
    return () => clearInterval(id)
  }, [token])

  if (!token) {
    return (
      <div className="shell">
        <h1>Tryzub Trade</h1>
        <p className="muted">React dashboard · Vite</p>
        <input value={user} onChange={(e) => setUser(e.target.value)} placeholder="username" />
        <input type="password" value={pass} onChange={(e) => setPass(e.target.value)} placeholder="password" />
        <button onClick={login}>Увійти</button>
        {err && <pre className="err">{err}</pre>}
      </div>
    )
  }

  const pnl = status?.risk?.pnl_pct ?? 0
  return (
    <div className="shell">
      <header>
        <h1>Tryzub Trade</h1>
        <span className="badge">{status?.testnet ? 'TESTNET' : 'MAINNET'}</span>
      </header>
      <section className="panel">
        <div className="muted">P&amp;L дня</div>
        <div className={`metric ${pnl >= 0 ? 'pos' : 'neg'}`}>{Number(pnl).toFixed(2)}%</div>
      </section>
      <section className="panel">
        <div className="muted">Features</div>
        <pre>{JSON.stringify(status?.features || {}, null, 2)}</pre>
        <div className="muted">Shadow / Tournament / Compound</div>
        <pre>
          {JSON.stringify(
            { shadow: status?.shadow, tournament: status?.tournament, compound: status?.compound },
            null,
            2,
          )}
        </pre>
      </section>
      <section className="panel">
        <button onClick={() => api('/api/scan', { method: 'POST' }).then(refresh)}>Скан</button>
        <button onClick={refresh}>Оновити</button>
      </section>
      <section className="panel">
        <div className="muted">Open</div>
        <table>
          <tbody>
            {(trades.open || []).map((t) => (
              <tr key={t.id}>
                <td>{t.symbol}</td>
                <td>{t.side}</td>
                <td>{t.entry_price}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </section>
      <section className="panel">
        <div className="muted">Closed</div>
        <table>
          <tbody>
            {(trades.closed || []).map((t) => (
              <tr key={t.id}>
                <td>{t.symbol}</td>
                <td>{t.pnl}</td>
                <td>{t.exit_reason}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </section>
      {err && <pre className="err">{err}</pre>}
    </div>
  )
}
