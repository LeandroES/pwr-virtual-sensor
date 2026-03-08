import { useState } from 'react'
import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import { Dashboard } from './pages/Dashboard'
import { VirtualSensorDashboard } from './pages/VirtualSensorDashboard'

const queryClient = new QueryClient({
  defaultOptions: {
    queries: {
      retry: 1,
      refetchOnWindowFocus: false,
      staleTime: 0,
    },
  },
})

type ActiveTab = 'physics' | 'sensor'

interface TabConfig {
  id:      ActiveTab
  label:   string
  sub:     string
  accent:  string
  border:  string
  dot:     string
}

const TABS: TabConfig[] = [
  {
    id:     'physics',
    label:  'Simulación PKE',
    sub:    'Radau · ScipySolver',
    accent: 'text-blue-400',
    border: 'border-blue-500',
    dot:    'bg-blue-500',
  },
  {
    id:     'sensor',
    label:  'Sensor Virtual',
    sub:    'EnKF · RK4 · GPU',
    accent: 'text-cyan-400',
    border: 'border-cyan-500',
    dot:    'bg-cyan-500',
  },
]

function TabNav({
  active,
  onChange,
}: {
  active: ActiveTab
  onChange: (tab: ActiveTab) => void
}) {
  return (
    <nav className="bg-slate-950 border-b border-slate-800">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        <div className="flex gap-0" role="tablist">
          {TABS.map((tab) => {
            const isActive = tab.id === active
            return (
              <button
                key={tab.id}
                role="tab"
                aria-selected={isActive}
                onClick={() => onChange(tab.id)}
                className={`
                  relative flex flex-col items-start px-5 py-3 text-left
                  transition-colors focus:outline-none focus-visible:ring-2 focus-visible:ring-cyan-500
                  ${isActive
                    ? `${tab.accent} border-b-2 ${tab.border}`
                    : 'text-slate-500 hover:text-slate-300 border-b-2 border-transparent'
                  }
                `}
              >
                <div className="flex items-center gap-2">
                  {isActive && (
                    <span className={`w-1.5 h-1.5 rounded-full shrink-0 ${tab.dot}`} />
                  )}
                  <span className="text-sm font-semibold font-mono tracking-wide">
                    {tab.label}
                  </span>
                </div>
                <span className="text-[10px] font-mono text-slate-600 pl-3.5">
                  {tab.sub}
                </span>
              </button>
            )
          })}
        </div>
      </div>
    </nav>
  )
}

export default function App() {
  const [activeTab, setActiveTab] = useState<ActiveTab>('physics')

  return (
    <QueryClientProvider client={queryClient}>
      <TabNav active={activeTab} onChange={setActiveTab} />
      {activeTab === 'physics' ? <Dashboard /> : <VirtualSensorDashboard />}
    </QueryClientProvider>
  )
}
