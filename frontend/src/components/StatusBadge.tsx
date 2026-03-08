import type { RunStatus } from '../types'

const STYLES: Record<RunStatus, string> = {
  pending:   'bg-amber-50  text-amber-800  border-amber-300',
  running:   'bg-blue-50   text-blue-800   border-blue-300',
  completed: 'bg-green-50  text-green-800  border-green-300',
  failed:    'bg-red-50    text-red-800    border-red-300',
}

const LABELS: Record<RunStatus, string> = {
  pending:   'Pendiente',
  running:   'Simulando…',
  completed: 'Completado',
  failed:    'Error',
}

interface Props {
  status: RunStatus
}

export function StatusBadge({ status }: Props) {
  return (
    <span
      className={`inline-flex items-center gap-1.5 px-3 py-1 rounded-full text-sm font-medium border ${STYLES[status]}`}
    >
      {status === 'running' && (
        <svg
          className="animate-spin h-3.5 w-3.5 shrink-0"
          fill="none"
          viewBox="0 0 24 24"
          aria-hidden="true"
        >
          <circle
            className="opacity-25"
            cx="12" cy="12" r="10"
            stroke="currentColor" strokeWidth="4"
          />
          <path
            className="opacity-75"
            fill="currentColor"
            d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4z"
          />
        </svg>
      )}
      {status === 'completed' && (
        <svg className="h-3.5 w-3.5 shrink-0" fill="none" viewBox="0 0 24 24" stroke="currentColor" aria-hidden="true">
          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2.5} d="M5 13l4 4L19 7" />
        </svg>
      )}
      {status === 'failed' && (
        <svg className="h-3.5 w-3.5 shrink-0" fill="none" viewBox="0 0 24 24" stroke="currentColor" aria-hidden="true">
          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2.5} d="M6 18L18 6M6 6l12 12" />
        </svg>
      )}
      {LABELS[status]}
    </span>
  )
}
