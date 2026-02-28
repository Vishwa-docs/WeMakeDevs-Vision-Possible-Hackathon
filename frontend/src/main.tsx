import React from 'react'
import { createRoot } from 'react-dom/client'
import './index.css'
import App from './App.tsx'

// ---------------------------------------------------------------------------
// Global Error Boundary — prevents blank pages on any uncaught render error
// ---------------------------------------------------------------------------
interface GEBState { error: Error | null }

class GlobalErrorBoundary extends React.Component<
  { children: React.ReactNode },
  GEBState
> {
  constructor(props: { children: React.ReactNode }) {
    super(props)
    this.state = { error: null }
  }
  static getDerivedStateFromError(error: Error) {
    return { error }
  }
  componentDidCatch(error: Error, info: React.ErrorInfo) {
    console.error('[WorldLens][FATAL] Uncaught render error:', error)
    console.error('[WorldLens][FATAL] Component stack:', info.componentStack)
  }
  render() {
    if (this.state.error) {
      return (
        <div
          style={{
            minHeight: '100vh',
            display: 'flex',
            flexDirection: 'column',
            alignItems: 'center',
            justifyContent: 'center',
            background: '#0a0a0f',
            color: '#e0e0e0',
            fontFamily: 'Inter, system-ui, sans-serif',
            padding: '2rem',
            textAlign: 'center',
          }}
        >
          <h1 style={{ fontSize: '2rem', marginBottom: '1rem' }}>
            🌍 WorldLens — Something went wrong
          </h1>
          <pre
            style={{
              background: '#1a1a2e',
              color: '#ff6b6b',
              padding: '1rem',
              borderRadius: '8px',
              maxWidth: '600px',
              overflow: 'auto',
              whiteSpace: 'pre-wrap',
              wordBreak: 'break-word',
              fontSize: '0.85rem',
              marginBottom: '1rem',
            }}
          >
            {this.state.error.message}
            {'\n\n'}
            {this.state.error.stack}
          </pre>
          <button
            onClick={() => {
              this.setState({ error: null })
              window.location.reload()
            }}
            style={{
              padding: '0.75rem 2rem',
              background: '#6c63ff',
              color: '#fff',
              border: 'none',
              borderRadius: '8px',
              cursor: 'pointer',
              fontSize: '1rem',
            }}
          >
            Reload App
          </button>
        </div>
      )
    }
    return this.props.children
  }
}

const root = createRoot(document.getElementById('root')!, {
  // React 19: catch errors that escape all error boundaries
  onUncaughtError: (error, errorInfo) => {
    console.error('[WorldLens][React] Uncaught error:', error, errorInfo)
  },
  onCaughtError: (error, errorInfo) => {
    console.error('[WorldLens][React] Caught error:', error, errorInfo)
  },
  onRecoverableError: (error, errorInfo) => {
    console.warn('[WorldLens][React] Recoverable error:', error, errorInfo)
  },
})

root.render(
  <GlobalErrorBoundary>
    <App />
  </GlobalErrorBoundary>,
)
