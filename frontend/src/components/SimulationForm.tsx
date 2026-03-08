import { useState, type FormEvent } from 'react'
import type { RunCreate } from '../types'

interface Props {
  onSubmit: (payload: RunCreate) => void
  isLoading: boolean
}

interface FieldProps {
  label: string
  hint: string
  value: number
  onChange: (v: number) => void
  step: number
  min: number
  max: number
  disabled: boolean
}

function Field({ label, hint, value, onChange, step, min, max, disabled }: FieldProps) {
  return (
    <div className="flex flex-col gap-1">
      <label className="text-sm font-medium text-gray-700">{label}</label>
      <input
        type="number"
        value={value}
        step={step}
        min={min}
        max={max}
        disabled={disabled}
        onChange={(e) => onChange(Number(e.target.value))}
        className="px-3 py-2 rounded-lg border border-gray-300 bg-white text-sm
                   focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent
                   disabled:bg-gray-100 disabled:text-gray-400 transition"
      />
      <p className="text-xs text-gray-400">{hint}</p>
    </div>
  )
}

export function SimulationForm({ onSubmit, isLoading }: Props) {
  // All user-facing values use "human" units; conversion happens on submit.
  const [reactivityPcm, setReactivityPcm] = useState<number>(100)   // pcm
  const [durationS, setDurationS]         = useState<number>(600)   // s
  const [dtS, setDtS]                     = useState<number>(1.0)   // s

  function handleSubmit(e: FormEvent<HTMLFormElement>) {
    e.preventDefault()
    onSubmit({
      external_reactivity: reactivityPcm * 1e-5,   // pcm → Δk/k
      time_span: [0.0, durationS],
      dt: dtS,
    })
  }

  return (
    <form onSubmit={handleSubmit} className="space-y-5">
      <div className="grid grid-cols-1 sm:grid-cols-3 gap-4">
        <Field
          label="Reactividad insertada (pcm)"
          hint="Rango típico: ±50 – 300 pcm  |  100 pcm = 1 × 10⁻³ Δk/k"
          value={reactivityPcm}
          onChange={setReactivityPcm}
          step={10}
          min={-500}
          max={500}
          disabled={isLoading}
        />
        <Field
          label="Duración de simulación (s)"
          hint="Tiempo hasta nuevo estado estacionario: ≥ 400 s"
          value={durationS}
          onChange={setDurationS}
          step={60}
          min={10}
          max={3600}
          disabled={isLoading}
        />
        <Field
          label="Paso de salida Δt (s)"
          hint="Paso del solver adaptativo es independiente de este valor"
          value={dtS}
          onChange={setDtS}
          step={0.5}
          min={0.1}
          max={60}
          disabled={isLoading}
        />
      </div>

      {/* Preview of derived values */}
      <div className="flex flex-wrap gap-3 text-xs text-gray-500">
        <span className="bg-gray-100 rounded px-2 py-1">
          ρ_ext = {(reactivityPcm * 1e-5).toExponential(2)} Δk/k
        </span>
        <span className="bg-gray-100 rounded px-2 py-1">
          Puntos estimados ≈ {Math.ceil(durationS / dtS) + 1}
        </span>
        <span className="bg-gray-100 rounded px-2 py-1">
          β ≈ 650 pcm → {Math.abs(reactivityPcm) < 650 ? 'Sub-prompt-crítico ✓' : 'Prompt-crítico ⚠'}
        </span>
      </div>

      <button
        type="submit"
        disabled={isLoading}
        className="inline-flex items-center gap-2 px-5 py-2.5 rounded-lg
                   bg-blue-600 hover:bg-blue-700 text-white text-sm font-semibold
                   disabled:opacity-50 disabled:cursor-not-allowed
                   transition-colors focus:outline-none focus:ring-2 focus:ring-blue-500 focus:ring-offset-2"
      >
        {isLoading ? (
          <>
            <svg className="animate-spin h-4 w-4" fill="none" viewBox="0 0 24 24">
              <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" />
              <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4z" />
            </svg>
            Simulando…
          </>
        ) : (
          <>
            <svg className="h-4 w-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2}
                d="M13 10V3L4 14h7v7l9-11h-7z" />
            </svg>
            Lanzar simulación
          </>
        )}
      </button>
    </form>
  )
}
