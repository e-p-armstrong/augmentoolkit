import { Routes, Route } from 'react-router-dom';
import { TaskHistoryProvider } from './context/TaskHistoryContext';
import { FileManagerProvider } from './context/FileManagerContext';
import Layout from './components/Layout';
import HomePage from './pages/HomePage';
import ConfigSelector from './pages/ConfigSelector';
import ConfigEditor from './pages/ConfigEditor';
import InputManager from './pages/InputManager';
import OutputManager from './pages/OutputManager';
import PipelineMonitor from './pages/PipelineMonitor';
import NotFound from './pages/NotFound';
import LogManager from './pages/LogManager';
import LogViewer from './pages/LogViewer';
import Chat from './pages/Chat';
// import './App.css'; // Keep or remove based on whether it has useful base styles

function App() {
  return (
    <TaskHistoryProvider>
      <FileManagerProvider>
        <Routes>
          <Route path="/" element={<Layout />}>
            <Route index element={<HomePage />} /> {/* Default route is now HomePage */}
            <Route path="configs" element={<ConfigSelector />} /> {/* Route for selecting configs */}
            <Route path="configs/:configId" element={<ConfigEditor />} /> {/* Route for editing a specific config */}
            <Route path="inputs" element={<InputManager />} />
            <Route path="outputs" element={<OutputManager />} />
            <Route path="monitor" element={<PipelineMonitor />} />
            <Route path="logs" element={<LogManager />} />
            <Route path="logs/:taskId" element={<LogViewer />} />
            <Route path="chat" element={<Chat />} />
            <Route path="*" element={<NotFound />} /> {/* Catch-all for 404 */}
          </Route>
        </Routes>
      </FileManagerProvider>
    </TaskHistoryProvider>
  );
}

export default App;
