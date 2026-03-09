import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
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

export default function App() {
  return (
    <QueryClientProvider client={queryClient}>
      <VirtualSensorDashboard />
    </QueryClientProvider>
  )
}
