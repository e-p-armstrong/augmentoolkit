import React, { useState, useEffect, useCallback, useRef, memo, useMemo } from 'react';
import { useSearchParams, Link, useNavigate } from 'react-router-dom'; // Import Link and useNavigate
import Editor from '@monaco-editor/react';
import { motion, AnimatePresence } from 'framer-motion';
import { FiCheckCircle, FiXCircle, FiClock, FiAlertCircle, FiRefreshCw } from 'react-icons/fi'; // Import icons
import * as api from '../api'; // Assuming api.js is in src/
import { saveAs } from 'file-saver'; // For downloading files
import { useTaskHistory } from '../context/TaskHistoryContext'; // Import the context hook
import IdleMonitorGraphic from '../components/IdleMonitorGraphic'; // Import the new graphic
import ProgressBarParticles from '../components/ProgressBarParticles'; // Import particle component
import EnhancedButton from '../components/EnhancedButton'; // Import the new button component

// --- Helper Functions (from ProgressBarParticles, could be moved to utils) ---
const interpolateColor = (color1, color2, factor) => {
    if (factor < 0) factor = 0;
    if (factor > 1) factor = 1;
    const c1 = parseInt(color1.substring(1), 16);
    const c2 = parseInt(color2.substring(1), 16);
    const r1 = (c1 >> 16) & 0xff, g1 = (c1 >> 8) & 0xff, b1 = c1 & 0xff;
    const r2 = (c2 >> 16) & 0xff, g2 = (c2 >> 8) & 0xff, b2 = c2 & 0xff;
    const r = Math.round(r1 + (r2 - r1) * factor);
    const g = Math.round(g1 + (g2 - g1) * factor);
    const b = Math.round(b1 + (b2 - b1) * factor);
    return `#${((1 << 24) + (r << 16) + (g << 8) + b).toString(16).slice(1).toUpperCase()}`;
};
const START_COLOR = '#4ade80'; // Dull Green (Tailwind green-400)
const END_COLOR = '#00FF00';   // Bright Neon Green
// --- End Helper Functions ---

// --- Helper Components (Wrap with React.memo) ---

const ProgressBar = memo(({ progress, message, status, progressBarRef, color }) => {
  const progressPercent = (progress * 100).toFixed(1);

  // Determine if the task is in an active, animating state
  const isActive = status === 'RUNNING' || status === 'PENDING';

  // --- Animation Calculations ---
  // Pulse intensity (scale variation): starts small, increases with progress
  const pulseIntensity = isActive ? 1 + (progress * 0.05) : 1; // Max scale 1.05 at 100%

  // Shake intensity (x/y offset): starts small, increases with progress
  const shakeIntensity = isActive ? progress * 1.5 : 0; // Max offset 1.5px at 100%

  // Glow intensity: brightens with progress
  const glowIntensity = isActive ? 4 + Math.round(progress * 8) : 4; // Glow from 4px to 12px

  // Helper for random numbers in animations
  const random = (min, max) => Math.random() * (max - min) + min;

  // Define animation variants/targets
  const barAnimations = {
      animate: {
          scale: isActive ? [1, pulseIntensity, 1] : 1, // Pulsate scale
          boxShadow: isActive ? [`0 0 ${glowIntensity}px ${color}`, `0 0 ${glowIntensity + 4}px ${color}`, `0 0 ${glowIntensity}px ${color}`] : `0 0 8px ${color}`, // Pulsate glow
          transition: isActive ? {
              scale: { duration: 1.5, repeat: Infinity, ease: "easeInOut", repeatType: "reverse" },
              boxShadow: { duration: 1.5, repeat: Infinity, ease: "easeInOut", repeatType: "reverse" },
              width: { duration: 0.5, ease: "easeOut" } // Keep width transition separate
          } : { duration: 0.5, ease: "easeOut" }
      },
      initial: { width: '0%' }, // Keep initial width for the bar fill
  };

  const containerAnimations = {
      animate: {
          x: isActive ? [0, random(-shakeIntensity, shakeIntensity), 0, random(-shakeIntensity, shakeIntensity), 0] : 0,
          y: isActive ? [0, random(-shakeIntensity, shakeIntensity), 0, random(-shakeIntensity, shakeIntensity), 0] : 0,
          transition: isActive ? {
              duration: 0.8, // Faster shake cycle
              repeat: Infinity,
              ease: "linear" // More erratic feel
          } : { duration: 0.5 }
      }
  };

  return (
    // Apply shaking animation to the outer container
    <motion.div
        className="w-full mb-4 relative" // Increased bottom margin
        variants={containerAnimations}
        animate="animate"
    >
      {/* Add ref to the outer container of the visible bar */}
      <div ref={progressBarRef} className="relative pt-1">
        {/* Make bar thicker */}
        <div className="overflow-hidden h-12 mb-1 text-xs flex rounded bg-gray-700 relative"> {/* Added relative positioning */}
          {/* Apply pulsing animation to the inner filled bar */}
          <motion.div
            // Use inline style for dynamic background color only
            style={{
                backgroundColor: color,
                // boxShadow is now handled by framer-motion animate
            }}
            // Removed flex items-center justify-center as the text is now outside this div
            className={`shadow-none transition-colors duration-500 ease-out`}
            initial={barAnimations.initial}
            animate={{ ...barAnimations.animate, width: `${progressPercent}%` }} // Combine width and other animations
            // Transitions are now defined within barAnimations.animate.transition
          >
            {/* Text moved outside the motion.div */}
           </motion.div>

          {/* Percentage Text - Centered relative to the whole bar */}
          {progress != null && ( // Only show text when progress is available
             <span className="absolute inset-0 flex items-center justify-center text-xl font-bold text-white pointer-events-none z-10">
                  {progressPercent}%
             </span>
          )}
        </div>
        <div className="flex justify-between items-center text-sm text-gray-400">
          <span>{status || 'Loading...'} {message || ''}</span>
          {/* Removed the percentage span from here */}
        </div>
      </div>
    </motion.div>
  );
});

const DetailsBox = memo(({ details }) => {
    const [isOpen, setIsOpen] = useState(false);
    const hasDetails = details && Object.keys(details).length > 0;

    if (!hasDetails) return null;

    return (
        <div className="mb-4 text-sm border border-gray-600 rounded p-2 bg-gray-800/50 max-w-md">
            <button onClick={() => setIsOpen(!isOpen)} className="text-left w-full font-semibold text-gray-300 mb-1">
                Details ({isOpen ? 'Hide' : 'Show'})
            </button>
            <AnimatePresence>
                {isOpen && (
                    <motion.div
                        initial={{ height: 0, opacity: 0 }}
                        animate={{ height: 'auto', opacity: 1 }}
                        exit={{ height: 0, opacity: 0 }}
                        transition={{ duration: 0.3 }}
                        className="overflow-hidden"
                    >
                        <ul className="list-disc list-inside pl-2 space-y-1 text-gray-400">
                            {Object.entries(details).map(([key, value]) => (
                                <li key={key}>
                                    <span className="font-medium text-gray-300">{key}:</span> {JSON.stringify(value)}
                                </li>
                            ))}
                        </ul>
                    </motion.div>
                )}
            </AnimatePresence>
        </div>
    );
});


const PendingTasks = memo(({ tasks, onCancel }) => {
    if (!tasks || tasks.length === 0) {
        return <div className="text-sm text-gray-500 mb-4">No pending tasks.</div>;
    }

    return (
        <div className="mb-4 p-2 border border-gray-600 rounded bg-gray-800/50">
            <h3 className="font-semibold text-gray-300 mb-2">Pending Queue ({tasks.length})</h3>
            <div className="flex flex-wrap gap-2">
                {tasks.map(taskId => (
                    <div key={taskId} className="group relative bg-gray-700 p-1 px-2 rounded text-xs text-gray-300 hover:bg-red-800 transition-colors">
                        <span className="truncate max-w-[150px] inline-block">{taskId}</span>
                         <button
                            onClick={() => onCancel(taskId)}
                            className="absolute inset-0 w-full h-full bg-red-600/80 text-white text-center opacity-0 group-hover:opacity-100 transition-opacity flex items-center justify-center text-xs font-bold rounded"
                            title={`Cancel Task ${taskId}`}
                         >
                            CANCEL
                         </button>
                    </div>
                ))}
            </div>
        </div>
    );
});

// Forward the ref to the underlying button element
const ActionButtons = memo(React.forwardRef(({ taskId, status, onCancel, onDownload }, ref) => {
  const isRunning = status === 'RUNNING';
  const isPending = status === 'PENDING';
  const canCancel = isRunning || isPending;
  
  const isCompleted = status === 'COMPLETED'

  // Define terminal states where download is suggested
  const isTerminalState = status === 'COMPLETED' || status === 'FAILED' || status === 'REVOKED';

  // Define button styles based on state
  let downloadButtonClass = 'px-4 py-2 rounded font-semibold text-white'; // Base classes for EnhancedButton wrapper
  let downloadParticleColor = '#3b82f6'; // Default blue

  if (!taskId) {
    // Disabled state handled by EnhancedButton's `disabled` prop
    downloadButtonClass += ' bg-gray-600'; // Keep background for disabled state
  } else if (isCompleted) {
    // Prominent/Recommended style for completed state
    downloadButtonClass += ' animate-pulse-green-download'; // Pulse when completed
    downloadParticleColor = '#22c55e'; // Green particles
  } else if (isTerminalState) {
    // Other terminal states (FAILED, REVOKED) - maybe different color?
    downloadButtonClass += ' bg-yellow-600'; // Yellow/Orange background?
    downloadParticleColor = '#f59e0b'; // Orange particles
  } else {
    // Active but less prominent style for non-terminal states (PENDING, RUNNING)
    downloadButtonClass += ' bg-blue-700 opacity-80'; // Dimmer blue, slightly transparent
    downloadParticleColor = '#60a5fa'; // Light blue particles
  }

  // Cancel button styles
  let cancelButtonClass = 'px-4 py-2 rounded font-semibold text-white';
  if (canCancel) {
    cancelButtonClass += ' bg-red-600';
  } else {
    cancelButtonClass += ' bg-gray-600'; // Disabled background
  }

  return (
    <div className="flex space-x-4">
      <EnhancedButton
        onClick={onCancel}
        disabled={!canCancel}
        className={cancelButtonClass}
        title={isRunning ? 'Cancel Running Task' : 'Revoke Pending Task'}
        particleColor="#ef4444" // Red particles for cancel/revoke
      >
        {isRunning ? 'Cancel Running Task' : 'Revoke Pending Task'}
      </EnhancedButton>
      <EnhancedButton
        onClick={onDownload}
        ref={ref} // Assign the forwarded ref here
        disabled={!taskId} // Only disable if no taskId
        className={downloadButtonClass} // Apply dynamic class
        title={isTerminalState ? "Download completed/failed task output" : "Download current task output (might be incomplete)"} // Dynamic title
        particleColor={downloadParticleColor} // Dynamic particle color
      >
        Download Output Folder
      </EnhancedButton>
    </div>
  );
}));


const ConfigViewer = memo(({ parameters }) => {
    const [isOpen, setIsOpen] = useState(true); // Start open?
    const configString = parameters ? JSON.stringify(parameters, null, 2) : "// No parameters loaded";

    // Determine language based on likely content (simple check)
    const language = typeof parameters === 'object' ? 'json' : 'plaintext';

     return (
        <div className="border border-gray-600 rounded bg-gray-800/50 h-full flex flex-col">
            <button
                onClick={() => setIsOpen(!isOpen)}
                className="text-left w-full font-semibold text-gray-300 p-2 border-b border-gray-600"
            >
                Config Used ({isOpen ? 'Hide' : 'Show'})
            </button>
            <AnimatePresence>
                {isOpen && (
                    <motion.div
                        initial={{ height: 0, opacity: 0 }}
                        animate={{ height: 'auto', opacity: 1 }}
                        exit={{ height: 0, opacity: 0 }}
                        transition={{ duration: 0.3 }}
                        className="overflow-hidden flex-grow p-1" // Added padding
                        style={{ minHeight: '100px' }} // Ensure minimum height
                    >
                        <Editor
                            height="100%" // Use available height
                            language={language}
                            value={configString}
                            theme="vs-dark" // Use a dark theme
                             options={{
                                readOnly: true,
                                minimap: { enabled: false },
                                scrollBeyondLastLine: false,
                                fontSize: 12,
                                wordWrap: 'on', // Wrap long lines
                             }}
                        />
                    </motion.div>
                )}
            </AnimatePresence>
        </div>
    );
});


const LogViewer = memo(({ logs, taskId, onDeleteLogs }) => {
    const logContainerRef = useRef(null);
    const [autoScroll, setAutoScroll] = useState(true);

    useEffect(() => {
        // If autoScroll is enabled and the ref is available, scroll to the bottom.
        if (autoScroll && logContainerRef.current) {
            // Directly set scrollTop to scrollHeight. This ensures it scrolls down
            // whenever logs change and autoScroll is true.
            logContainerRef.current.scrollTop = logContainerRef.current.scrollHeight;
        }
        // No dependency on clientHeight/scrollTop needed here, only on logs changing
        // and the autoScroll toggle state.
    }, [logs, autoScroll]); // Re-run this effect when logs content changes or autoScroll is toggled

    const handleScroll = () => {
        if (logContainerRef.current) {
            const { scrollTop, scrollHeight, clientHeight } = logContainerRef.current;
            // Disable autoscroll if user scrolls up significantly
             if (scrollHeight - scrollTop > clientHeight + 20) {
                 setAutoScroll(false);
             } else {
                 setAutoScroll(true); // Re-enable if scrolled back to bottom
             }
        }
    };

    return (
        <div className="border border-gray-600 rounded bg-gray-800/50 flex flex-col h-[400px]"> {/* Fixed height */}
            <div className="flex justify-between items-center p-2 border-b border-gray-600">
                <h3 className="font-semibold text-gray-300">Task Logs</h3>
                 <div className="flex items-center space-x-4">
                    <label className="flex items-center space-x-1 text-xs text-gray-400 cursor-pointer">
                         <input type="checkbox" checked={autoScroll} onChange={() => setAutoScroll(s => !s)} className="form-checkbox h-3 w-3 text-blue-500 bg-gray-700 border-gray-600 rounded focus:ring-blue-600"/>
                         <span>Autoscroll</span>
                     </label>
                    <button
                        onClick={onDeleteLogs}
                        disabled={!logs || !taskId}
                        title="Delete Logs"
                        className={`px-2 py-1 text-xs rounded font-semibold text-white transition-colors duration-150 ease-in-out
                                    ${logs && taskId ? 'bg-red-700 hover:bg-red-800 active:bg-red-900'
                                                : 'bg-gray-600 cursor-not-allowed opacity-50'}`}
                    >
                        Clear Logs
                    </button>
                 </div>
            </div>
            <div
                ref={logContainerRef}
                onScroll={handleScroll}
                className="flex-grow overflow-auto p-2 font-mono text-xs text-gray-300 whitespace-pre-wrap break-words" // Ensure wrapping and preserve format
            >
                {logs || "No logs yet..."}
            </div>
        </div>
    );
});

// --- New Task History List Component ---
const TaskHistoryList = memo(({ history, currentTaskId, onSelectTask, taskStatuses, onClearHistory }) => {
    console.log(`[TaskHistoryList Render] History prop:`, history);
    if (!history || history.length === 0) {
        return (
            <div className="mt-8 p-4 border border-gray-600 rounded bg-gray-800/50">
                <div className="flex justify-between items-center mb-3">
                    <h3 className="text-lg font-semibold text-gray-300">Session Task History</h3>
                    {history && history.length > 0 && (
                        <button
                            onClick={() => onClearHistory()}
                            className="text-xs px-2 py-1 rounded bg-red-800 hover:bg-red-700 text-red-100 transition-colors"
                        >
                            Clear History
                        </button>
                    )}
                </div>
                <p className="text-sm text-gray-500">No tasks run or viewed this session.</p>
            </div>
        );
    }

    const getStatusIcon = (status) => {
        switch (status) {
            case 'COMPLETED': return <FiCheckCircle className="text-green-500" />;
            case 'FAILED': return <FiXCircle className="text-red-500" />;
            case 'RUNNING': return <FiRefreshCw className="animate-spin text-blue-500" />;
            case 'PENDING': return <FiClock className="text-yellow-500" />;
            case 'REVOKED': return <FiAlertCircle className="text-yellow-600" />;
            default: return <FiClock className="text-gray-500" />; // Default/Unknown
        }
    };

    const handleClear = () => {
        if (window.confirm('Are you sure you want to clear the task history? This cannot be undone.')) {
            onClearHistory();
        }
    };

    return (
        <div className="mt-8 p-4 border border-gray-600 rounded bg-gray-800/50">
            <div className="flex justify-between items-center mb-3">
                <h3 className="text-lg font-semibold text-gray-300">Session Task History</h3>
                {history && history.length > 0 && (
                    <button
                        onClick={handleClear}
                        className="text-xs px-2 py-1 rounded bg-red-800 hover:bg-red-700 text-red-100 transition-colors"
                    >
                        Clear History
                    </button>
                )}
            </div>
            <ul className="space-y-2 max-h-60 overflow-y-auto pr-2">
                {history.map(taskId => {
                    const isActive = taskId === currentTaskId;
                    const statusInfo = taskStatuses[taskId] || { status: 'Loading...' };
                    return (
                        <li key={taskId}>
                            <Link
                                to={`/monitor?pipeline_id=${taskId}`}
                                onClick={(e) => { e.preventDefault(); onSelectTask(taskId); }}
                                className={`flex items-center justify-between p-2 rounded transition-colors duration-150 ease-in-out group ${isActive ? 'bg-blue-900/50 cursor-default' : 'bg-gray-700 hover:bg-gray-600'}`}
                            >
                                <span className={`font-mono text-sm truncate ${isActive ? 'text-blue-300 font-medium' : 'text-gray-300 group-hover:text-white'}`}>
                                    {taskId}
                                </span>
                                <span className="flex items-center space-x-1 text-xs ml-2 flex-shrink-0">
                                     {getStatusIcon(statusInfo.status)}
                                     <span className={isActive ? 'text-blue-400' : 'text-gray-400'}>{statusInfo.status}</span>
                                </span>
                            </Link>
                        </li>
                    );
                })}
            </ul>
        </div>
    );
});


// --- Main Monitor Component ---

function PipelineMonitor() {
  const [searchParams, setSearchParams] = useSearchParams(); // Use setSearchParams
  const navigate = useNavigate(); // Use navigate hook
  const urlTaskId = searchParams.get('pipeline_id');
  const { taskHistory, addTaskToHistory, clearTaskHistory } = useTaskHistory(); // Use context

  const [taskId, setTaskId] = useState(urlTaskId || null);
  const [status, setStatus] = useState(null);
  const [progress, setProgress] = useState(null);
  const [message, setMessage] = useState('');
  const [details, setDetails] = useState(null);
  const [parameters, setParameters] = useState(null);
  const [logs, setLogs] = useState('');
  const [pendingTasks, setPendingTasks] = useState([]);
  const [error, setError] = useState(null);
  const [isLoading, setIsLoading] = useState(false);
  const [showStartupHint, setShowStartupHint] = useState(false);

  const [historyTaskStatuses, setHistoryTaskStatuses] = useState({}); // Store statuses for history list

  // Ref for the progress bar element
  const progressBarRef = useRef(null);
  // Ref for the download button element
  const downloadButtonRef = useRef(null);

  // Use refs to store previous values for comparison to avoid unnecessary state updates
  const prevStatusRef = useRef();
  const prevProgressRef = useRef();
  const prevMessageRef = useRef();
  const prevDetailsRef = useRef();
  const prevPendingTasksRef = useRef();
  const prevLogsRef = useRef();

  useEffect(() => {
      // Update refs whenever state changes
      prevStatusRef.current = status;
      prevProgressRef.current = progress;
      prevMessageRef.current = message;
      prevDetailsRef.current = details;
      prevPendingTasksRef.current = pendingTasks;
      prevLogsRef.current = logs;
  }); // Runs after every render

  const logsTailLength = 500; // How many lines to fetch initially/periodically

  // --- API Callbacks ---

  const fetchStatus = useCallback(async () => {
    if (!taskId) return;
    try {
      const data = await api.fetchTaskStatus(taskId);

      // Only update state if data has actually changed
      let changed = false;
      if (data.status !== prevStatusRef.current) {
          setStatus(data.status);
          changed = true;
      }
      // Check progress specifically for null/undefined cases and actual value changes
      const currentProgress = prevProgressRef.current;
      const nextProgress = data.progress;
      if (nextProgress !== undefined && nextProgress !== null && nextProgress !== currentProgress) {
          setProgress(nextProgress);
          changed = true;
      }
      if (data.message !== prevMessageRef.current) {
          setMessage(data.message || '');
          changed = true;
      }
      // Simple stringify for details comparison (adjust if needed for deep comparison)
      if (JSON.stringify(data.details) !== JSON.stringify(prevDetailsRef.current)) {
          setDetails(data.details || null);
          changed = true;
      }

      // Clear error only if we successfully fetched (regardless of data change)
      setError(null);

      // Optional: Log if data was fetched but no change detected
      // if (!changed) { console.log('Status fetched, no change.'); }

    } catch (err) {
      console.error("Error fetching status:", err);
       // Avoid setting error repeatedly for the same issue
       const errorMsg = `Failed to fetch status: ${err.message}`;
       if (error !== errorMsg) {
           setError(errorMsg);
       }
      if (err.status === 404) {
        if (status !== 'NOT_FOUND') {
            setStatus('NOT_FOUND');
            setMessage(`Task ${taskId} not found.`);
        }
      }
    }
  }, [taskId]); // Use refs for comparison, only need taskId to know *which* task to fetch

  const fetchParams = useCallback(async () => {
    if (!taskId || parameters) return; // Keep the check for existing parameters
    setIsLoading(true);
    try {
      const data = await api.fetchTaskParameters(taskId);
      setParameters(data.parameters); // Can be null if not found/expired
      // setError(null); // Remove error clearing here, let status fetch handle it
    } catch (err) {
      console.error("Error fetching parameters:", err);
      // Only set main error if it's NOT a 404
      if (err.status !== 404) {
          const errorMsg = `Failed to fetch parameters: ${err.message}`;
          if (error !== errorMsg) setError(errorMsg);
      }
      // If it IS a 404, we just leave parameters null/unchanged
      // Set parameters to null explicitly on any error to clear previous ones?
      // setParameters(null);
    } finally {
        setIsLoading(false);
    }
  }, [taskId]); // Removed 'parameters' and 'error' dependencies

  const fetchLogs = useCallback(async () => {
      if (!taskId) return;
      try {
          const data = await api.fetchTaskLogs(taskId, logsTailLength);
          // Use ref for comparison
          if (data.logs !== prevLogsRef.current) {
             setLogs(data.logs);
             // setError(null); // Clear error on successful log update - Keep this?
          } else {
             // console.log('Logs fetched, no change.');
          }
      } catch (err) {
          console.error("Error fetching logs:", err);
           if (err.status === 404) {
              // On initial 404, show "Waiting...", don't set main error
              if (logs === '' || logs === 'No logs yet...' || logs === 'Waiting for logs...') { // Check if logs are in an initial/empty state
                  setLogs('Waiting for logs...'); 
              }
              // Don't set the main error state for 404s
           } else {
                // For other errors, maybe show a transient error or log it
                const errorMsg = `Failed to fetch logs: ${err.message}`;
                // if (error !== errorMsg) {
                //      setError(errorMsg); // Decide if log fetch errors should pollute main error state
                // }
                console.warn(`Non-404 error fetching logs: ${errorMsg}`); // Log non-404 errors
           }
      }
  }, [taskId, logsTailLength, logs]); // Added 'logs' back as dependency to check initial state


  const fetchQueue = useCallback(async () => {
    try {
      const data = await api.fetchTaskQueue();
      const currentPendingString = JSON.stringify(prevPendingTasksRef.current);
      const nextPendingString = JSON.stringify(data.pending_tasks || []);

      if (nextPendingString !== currentPendingString) {
        setPendingTasks(data.pending_tasks || []);
        setError(null); // Clear error on successful queue update
      } else {
        // console.log('Queue fetched, no change.');
      }
    } catch (err) { // Only set error if it changes
      console.error("Error fetching queue:", err);
      const errorMsg = `Failed to fetch queue: ${err.message}`;
      if (error !== errorMsg) {
          setError(errorMsg);
      }
    }
  }, []); // Reads pending tasks via ref, clears error on success, sets error on fail - no dependency needed

  const handleCancelTask = useCallback(async (idToCancel) => {
    const confirmMessage = idToCancel === taskId
        ? `Are you sure you want to cancel the currently monitored task (${taskId})?`
        : `Are you sure you want to cancel pending task ${idToCancel}?`;

    if (window.confirm(confirmMessage)) {
      setIsLoading(true);
      setError(null);
      try {
        const result = await api.interruptTask(idToCancel);
        console.log("Interrupt/Revoke result:", result);
        // Refresh status and queue immediately
        if (idToCancel === taskId) fetchStatus();
        fetchQueue();
        // Optionally show a success message
      } catch (err) {
        console.error(`Error interrupting/revoking task ${idToCancel}:`, err);
        setError(`Failed to cancel task ${idToCancel}: ${err.message}`);
      } finally {
        setIsLoading(false);
      }
    }
  }, [taskId, fetchStatus, fetchQueue]);

 const handleDownloadOutput = useCallback(async () => {
     if (!taskId) return;
     setIsLoading(true);
     setError(null);
     try {
         const response = await api.downloadTaskOutput(taskId);
         const blob = await response.blob();
         // Determine filename (e.g., from Content-Disposition or default)
         let filename = `task_${taskId}_output.zip`;
         const disposition = response.headers.get('content-disposition');
         if (disposition && disposition.indexOf('attachment') !== -1) {
             const filenameRegex = /filename[^;=\n]*=((['"]).*?\2|[^;\n]*)/;
             const matches = filenameRegex.exec(disposition);
             if (matches != null && matches[1]) {
                 filename = matches[1].replace(/['"]/g, '');
             }
         }
         saveAs(blob, filename); // Use file-saver library
     } catch (err) {
         console.error(`Error downloading output for task ${taskId}:`, err);
         setError(`Download failed: ${err.message}`);
     } finally {
         setIsLoading(false);
     }
 }, [taskId]);


  const handleDeleteLogs = useCallback(async () => {
    if (!taskId) return;
    if (window.confirm(`Are you sure you want to delete the logs for task ${taskId}? This cannot be undone.`)) {
      setIsLoading(true);
      setError(null);
      try {
        const result = await api.deleteTaskLogs(taskId);
        console.log("Delete logs result:", result);
        setLogs(''); // Clear logs locally
        // Optionally show success
      } catch (err) {
        console.error(`Error deleting logs for task ${taskId}:`, err);
        setError(`Failed to delete logs: ${err.message}`);
      } finally {
        setIsLoading(false);
      }
    }
  }, [taskId]);

  // --- Helper function for fetching history statuses ---
  const fetchHistoryStatuses = useCallback(async () => {
    const statusesToFetch = taskHistory.filter(id => {
        const current = historyTaskStatuses[id];
        // Fetch if not present, or if status is potentially outdated (not final)
        return !current || !['COMPLETED', 'FAILED', 'REVOKED', 'NOT_FOUND'].includes(current.status);
    });

    if (statusesToFetch.length === 0) {
        // console.log("No non-final history statuses to update.");
        return; // Nothing to fetch
    }

    // console.log("Fetching statuses for history:", statusesToFetch);
    const promises = statusesToFetch.map(id =>
        api.fetchTaskStatus(id).catch(err => ({
            task_id: id,
            status: err.status === 404 ? 'NOT_FOUND' : 'ERROR',
            message: err.message
        }))
    );

    try {
        const results = await Promise.all(promises);
        setHistoryTaskStatuses(prevStatuses => {
            const newStatuses = { ...prevStatuses };
            let hasChanged = false;
            results.forEach(statusResult => {
                if (statusResult && statusResult.task_id) {
                    const existingStatus = prevStatuses[statusResult.task_id]?.status;
                    if (statusResult.status !== existingStatus) {
                        newStatuses[statusResult.task_id] = { status: statusResult.status };
                        hasChanged = true;
                    }
                }
            });
            // Only update state if something actually changed to prevent unnecessary re-renders
            return hasChanged ? newStatuses : prevStatuses;
        });
    } catch (error) {
        // This catch might not be strictly necessary due to the individual catches in map,
        // but good for safety.
        console.error("Error processing history status results:", error);
    }
  }, [taskHistory, historyTaskStatuses]); // Keep dependencies

  // --- Effects --- // Renamed section slightly

  // Effect 1: Update taskId from URL and reset state when it changes
  useEffect(() => {
    const currentUrlTaskId = searchParams.get('pipeline_id');
    console.log(`[PipelineMonitor Effect 1] URL Task ID: ${currentUrlTaskId}, Current State Task ID: ${taskId}`);

    // Check if the relevant part of searchParams actually changed the target taskId
    if (currentUrlTaskId !== taskId) {
      console.log(`[PipelineMonitor Effect 1] Task ID changed from ${taskId} to ${currentUrlTaskId}. Resetting state and attempting to add to history.`);
      setTaskId(currentUrlTaskId);

      // Reset state for the new/cleared task
      setStatus(null);
      setProgress(null);
      setMessage('');
      setDetails(null);
      setParameters(null); // Crucial: Ensure parameters are reset to trigger refetch
      setLogs('');
      setError(null);
      setIsLoading(!!currentUrlTaskId); // Set loading only if there's a new task ID
    }

    // Add the task to history (only if it's a valid new ID)
    if (currentUrlTaskId) {
        console.log(`[PipelineMonitor Effect 1] Calling addTaskToHistory with: ${currentUrlTaskId}`);
        addTaskToHistory(currentUrlTaskId);
    }

  }, [searchParams, taskId, addTaskToHistory]); // Ensure all dependencies are listed


  // Effect 2: Fetch initial data when taskId changes
  useEffect(() => {
    if (!taskId) {
      console.log("[PipelineMonitor Effect 2] No Task ID, skipping initial fetch.");
      // Ensure loading is false if there's no task ID
      setIsLoading(false);
      return; // Don't fetch if no task ID
    }

    console.log(`[PipelineMonitor Effect 2] Task ID ${taskId} detected, fetching initial data.`);
    // We have a taskId, set loading and fetch initial data
    setIsLoading(true);
    const fetchInitialData = async () => {
      try {
        console.log(`[PipelineMonitor Effect 2] Fetching status for ${taskId}...`);
        await fetchStatus(); // Fetch status first
        console.log(`[PipelineMonitor Effect 2] Fetching parameters for ${taskId}...`);
        await fetchParams(); // Then params
        console.log(`[PipelineMonitor Effect 2] Fetching logs for ${taskId}...`);
        await fetchLogs();   // Then logs
        console.log(`[PipelineMonitor Effect 2] Initial data fetch complete for ${taskId}.`);
        // Fetch params, status, and logs concurrently
        // await Promise.all([fetchParams(), fetchStatus(), fetchLogs()]);
        // Clear any previous error if all fetches succeed initially
        // setError(null); // fetchStatus/fetchLogs already handle clearing error on success
      } catch (err) {
        // Errors should be handled within individual fetch functions now
        console.error("[PipelineMonitor Effect 2] Unexpected error during sequential initial data fetch:", err);
        // setError(`Failed during initial data sequence for ${taskId}. See console.`); // Optionally set a generic error
      } finally {
        // Important: Set loading false *after* initial fetches complete
        console.log(`[PipelineMonitor Effect 2] Setting isLoading to false for ${taskId}.`);
        setIsLoading(false);
      }
    };

    fetchInitialData();

    // No cleanup needed here, this effect just runs the initial fetch

  }, [taskId, fetchParams, fetchStatus, fetchLogs]); // Runs when taskId or fetch functions change


  // Effect 3: Polling for Status and Logs based on taskId and status
  useEffect(() => {
    // Determine if polling should be active
    const isTaskActive = taskId && status && !['COMPLETED', 'FAILED', 'REVOKED', 'NOT_FOUND'].includes(status);

    if (!isTaskActive) {
      return; // Stop polling if task inactive/finished/not found/no status yet
    }

    // Set up polling intervals
    const statusInterval = setInterval(fetchStatus, 2000); // Poll status every 2s
    const logsInterval = setInterval(fetchLogs, 3000);   // Poll logs every 3s

    // Cleanup function: Clear intervals when taskId or status changes, or component unmounts
    return () => {
      clearInterval(statusInterval);
      clearInterval(logsInterval);
    };

  }, [taskId, status, fetchStatus, fetchLogs]); // Re-run when taskId, status, or fetch functions change


  // Effect 4: Polling for Queue (Independent)
  useEffect(() => {
    fetchQueue(); // Initial fetch
    const queueInterval = setInterval(fetchQueue, 10000); // Poll queue every 10s
    return () => clearInterval(queueInterval);
  }, [fetchQueue]);


  // Effect 5: Polling for Task History Statuses (Independent)
  useEffect(() => {
    fetchHistoryStatuses(); // Fetch immediately
    const historyInterval = setInterval(fetchHistoryStatuses, 10000); // Poll every 10 seconds
    return () => clearInterval(historyInterval);
  }, [fetchHistoryStatuses]); // Re-run if the callback changes

  // Effect to show a hint on task load, then hide it.
  useEffect(() => {
    let timer;
    if (taskId) {
        setShowStartupHint(true);
        timer = setTimeout(() => {
            setShowStartupHint(false);
        }, 20000); // 20 seconds
    } else {
        setShowStartupHint(false);
    }
    return () => {
        if (timer) clearTimeout(timer);
    }
  }, [taskId]);

  // --- Helper Functions ---
  const handleSelectHistoryTask = (selectedTaskId) => {
    // Update the URL query parameter, which triggers the main useEffect to load the task
    setSearchParams({ pipeline_id: selectedTaskId });
  };

  // --- Render Logic ---

  // Calculate the dynamic color for the progress bar and particles
  const dynamicColor = useMemo(() => {
      if (status === 'FAILED') {
          return '#EF4444'; // Red
      } else if (status === 'REVOKED') {
          return '#F59E0B'; // Orange
      } else if (status === 'COMPLETED') {
          return END_COLOR; // Bright Neon Green
      } else {
          // Interpolate for PENDING/RUNNING or null status
          return interpolateColor(START_COLOR, END_COLOR, progress ?? 0);
      }
  }, [status, progress]);

  return (
    // Added overflow-y-auto to the main container if content might exceed viewport
    <div className="relative p-8 text-gray-300 bg-gray-900 min-h-screen flex flex-col overflow-y-auto">
       {/* MOVED Particle Background back here AND pass taskId */}
       <ProgressBarParticles
         taskId={taskId} // Pass taskId down
         progress={progress}
         status={status}
         progressBarElement={progressBarRef.current} // Pass the DOM element
         downloadButtonElement={downloadButtonRef.current} // Pass the download button DOM element
         color={dynamicColor} // Pass the calculated color
       />
       {/* Content needs z-index to be above particles */}
       <div className="relative z-10 flex-grow flex flex-col"> {/* Added flex flex-col here */}
          {/* Conditional Rendering based on taskId */}
          {!taskId ? (
            // Render view when no task is selected
            <div className="p-4 text-gray-400 flex flex-col items-center justify-center h-full"> {/* Center content */}
                 <h1 className="text-xl font-semibold mb-4 text-center">Pipeline Monitor</h1>
                 {/* Add the Idle Graphic */}
                <div className="my-8">
                    <IdleMonitorGraphic />
                </div>
                <p className="text-center">No pipeline task selected. Select a task from the history below, <br /> or run a new pipeline from the Configs page.</p>
                 <div className="mt-6 max-w-md w-full"> {/* Ensure pending tasks fit nicely */}
                     <PendingTasks tasks={pendingTasks} onCancel={handleCancelTask} />
                 </div>
            </div>
          ) : (
            // Render view when a task IS selected
            <>
              {/* REMOVED Particle Background from here */}
              <div className="flex flex-col flex-grow mb-12"> {/* Ensure this div takes height */}
                   <h1 className="text-2xl font-bold mb-1">Pipeline Monitor: <span className="font-mono text-lg text-blue-400">{taskId}</span></h1>
                    {/* Progress Bar - pass the ref */}
                    <ProgressBar progressBarRef={progressBarRef} progress={progress ?? 0} message={message} status={status} color={dynamicColor} />
                    {/* Details & Pending Tasks (removed wrapper) */}
                     {/* <div className="flex flex-wrap gap-4 mb-4">
                        <DetailsBox details={details} />
                        <PendingTasks tasks={pendingTasks} onCancel={handleCancelTask} />
                    </div> */}

                    {/* Hint for pipeline not starting */}
                    <AnimatePresence>
                        {showStartupHint && (
                            <motion.div
                                initial={{ opacity: 0 }}
                                animate={{ opacity: 1 }}
                                exit={{ opacity: 0 }}
                                transition={{ duration: 0.5 }}
                                className="mt-2 text-xs text-gray-400 p-2 bg-gray-800/50 border border-gray-600 rounded"
                            >
                                Hint: If it seems like a pipeline has not started after a medium amount of time, but you see no error, try cancelling it and re-running it with the same config.
                            </motion.div>
                        )}
                    </AnimatePresence>

                     {/* Error Display */}
                    {error && (
                        <div className="my-4 p-3 bg-red-800/50 border border-red-700 rounded text-red-200">
                            <strong>Error:</strong> {error}
                        </div>
                    )}
                    {isLoading && (
                        <div className="fixed top-4 right-4 z-50 p-2 bg-blue-600 text-white text-xs rounded shadow-lg">Loading...</div>
                    )}
                    {/* Main Content Area - Adjusted grid for alignment */}
                    <div className="mt-12 flex-grow grid grid-cols-1 md:grid-cols-3 gap-6 mb-6"> {/* Add flex-grow */}
                        {/* Left Column: Actions & Config - Make it flex col to fill height */}
                        <div className="md:col-span-1 flex flex-col gap-6">
                            <div className="p-4 border border-gray-600 rounded bg-gray-800/50 flex-shrink-0"> {/* Actions box doesn't grow */}
                                <h3 className="text-lg font-semibold mb-3 text-gray-300">Actions</h3>
                                <ActionButtons
                                    taskId={taskId}
                                    ref={downloadButtonRef} // Pass the ref here
                                    status={status}
                                    onCancel={() => handleCancelTask(taskId)}
                                    onDownload={handleDownloadOutput}
                                />
                                <p className="mt-3 text-xs text-gray-500">
                                    Hint: If cancelling a running task seems to have not worked, wait a few seconds -- the background process is probably cleaning up.
                                </p>
                                <p className="mt-3 text-xs text-gray-500">
                                  Also, while things should be cleaned up, be sure to double-check whether your pod has been correctly deleted after training is finished on Runpod.
                                </p>
                            </div>
                            {/* Config viewer grows to fill remaining space */}
                             <div className="flex-grow min-h-0"> {/* Added min-h-0 here */}
                                 <ConfigViewer parameters={parameters} />
                             </div>
                        </div>
                         {/* Right Column: Logs - Should have fixed height */}
                        <div className="md:col-span-2">
                            {/* LogViewer already has fixed height h-[400px] */}
                            <LogViewer logs={logs} taskId={taskId} onDeleteLogs={handleDeleteLogs} />
                        </div>
                    </div>
              </div>
            </>
          )}
       </div>

        
       <hr className='w-48 mx-auto border-gray-700'/>

        {/* Task History List (Always visible at the bottom) */}
        {/* <hr className="my-4 border-t-2 border-gray-700 -z-20" /> */}
        <div className="relative z-10 mt-auto pt-4"> {/* Pushes history to the bottom, ensure z-index */} 
             <TaskHistoryList
                 history={taskHistory}
                 currentTaskId={taskId}
                 onSelectTask={handleSelectHistoryTask}
                 taskStatuses={historyTaskStatuses}
                 onClearHistory={clearTaskHistory}
             />
        </div>
    </div>
  );
}

export default PipelineMonitor; 


// this page has the elements:
// the progress bar (top, across entire screen. Displays colored progress % based on the progress value, and displays beneath it with a subtitle the message of the pipeline (this element is basically goverened by GET tasks/taskid/status). Also shows the status somehow.)
// Beneath the progress bar -- a list of the pipelines which are currently queued up (/tasks/queue) with buttons to cancel any of them before they start (.../interrupt). Hover visual effect with click to cancel subtitle over them that appears when you hover. The list is... how to say this, horizontal? Like it is not done like a file thing where it's one item per bit of horizontal space, there's a bit of vertical space instead, and it will wrap if it has to. 
// a details box that shows each key/value from the details part of the response. If it exists. Otherwise the box is not present or is empty.
// the buttons for interacting with it (cancel, download output folder using task ID). In a box to the left
// a little expandable window with a (read only) monaco editor for the config. We can go to its config in the config folder as well (if it is in fact located in external_configs). In a box/little deliniated area to the right.
// Sketch of the page in ascii art


// the top navigation (you don't need to recreate this, this is being rendered inside that already)
// ----------------------------
// |           xy.z%          | <--- progress bar extends across whole screen. governed by GET tasks/taskid/status
// ----------------------------
// message subtitle
//
// Details (click to expand) (only present if details actually has stuff)
// -------
// key: value
// ...
// make this box economical on vertical space if possible, smaller text, it will only look good if the pending pipelines are close to the actual progress bar so this has to be on the smaller side
// -------
//
// pending pipeline 1 | pending pipeline 2 | pending pipeline 3 | ... (click to cancel on hover)
// ^ governed by GET tasks/queue
// ————————————————————————————              || —————————————————————————————————
// --------      -------                     || Config viewer (read only monaco text editor showing the config. We fetch it once with the new GET /tasks/taskid/parameters).
// | CANCEL|     | DOWNLOAD OUTPUT FOLDER |  ||
// |_______|     |______|                    ||
// (relatively big buttons. Not HUGE, but large. They are the primary actions on this page after all)                                         ||
//-----------------------------              || ------------------------------------
//
// logs viewer for this task
// -----------------------------------------------------------
// | a big box that shows the logs for this task as they progress. Uses /tasks/taskid/logs and uses the tail parameter + appending the response to what it has already, to avoid fetching the entire thing each refresh |
// -----------------------------------------------------------
// DELETE LOGS BUTTON (has confirmation modal, points DELETE tasks/taskid/logs)
