import { StrictMode } from 'react'
import { createRoot } from 'react-dom/client'
import { BrowserRouter } from 'react-router-dom'
import { NavigationProvider } from './context/NavigationContext'
import './index.css'
import App from './App.jsx'

createRoot(document.getElementById('root')).render(
  <StrictMode>
    <BrowserRouter>
      <NavigationProvider>
        <App />
      </NavigationProvider>
    </BrowserRouter>
  </StrictMode>,
)
