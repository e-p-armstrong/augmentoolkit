/* ------------------------------------------------------------------ */
/*  Imports + stable placeholder regex                                */
/* ------------------------------------------------------------------ */
import React, { useState, useEffect, useCallback, useRef } from 'react';
import { useParams, useNavigate, Link } from 'react-router-dom';
import Editor from '@monaco-editor/react';
import yaml from 'js-yaml';
import { FiChevronLeft, FiX, FiLoader } from 'react-icons/fi';
import Modal from '../components/Modal';
import EnhancedButton from '../components/EnhancedButton';
import {
  fetchConfigContent,
  saveConfigContent,
  fetchAvailablePipelines,
  runPipeline,
} from '../api';

/* 1️⃣  Define the regex **outside** the component so it never changes */
const PLACEHOLDER_REGEX = /!!PLACEHOLDER!!/;
const ATTENTION_REGEX = /!!ATTENTION!!/;

/* ------------------------------------------------------------------ */
/*                              ConfigEditor                          */
/* ------------------------------------------------------------------ */
function ConfigEditor() {
  /* ---------- routing & refs ---------- */
  const { configId: encodedConfigId } = useParams();
  const configId = decodeURIComponent(encodedConfigId || '');
  const navigate = useNavigate();
  const editorRef = useRef(null);
  const monacoRef = useRef(null);
  const placeholderDecorationsRef = useRef([]);
  const attentionDecorationsRef = useRef([]);

  /* ---------- editor state ---------- */
  const [originalContent, setOriginalContent] = useState('');
  const [currentContent, setCurrentContent] = useState('');
  const [isLoading, setIsLoading] = useState(true);
  const [isSaving, setIsSaving] = useState(false);
  const [editorError, setEditorError] = useState(null);
  const [saveSuccess, setSaveSuccess] = useState(false);
  const isDirty = originalContent !== currentContent;

  /* ---------- placeholder/attention detection ---------- */
  const [placeholderLines, setPlaceholderLines] = useState([]); // line numbers (1-based)
  const [attentionLines, setAttentionLines] = useState([]); // line numbers (1-based)

  /** Regex that matches: `key: !!PLACEHOLDER!!`, `key: '!!PLACEHOLDER!!'`, `key: "!!PLACEHOLDER!!"` */
  // const PLACEHOLDER_REGEX = /:\s*['"]?!!PLACEHOLDER!!['"]?\s*$/; // Defined outside now

  /* ---------- run & modal state ---------- */
  const [isRunModalOpen, setIsRunModalOpen] = useState(false);
  const [availablePipelines, setAvailablePipelines] = useState([]);
  const [isFetchingPipelines, setIsFetchingPipelines] = useState(false);
  const [modalNodePath, setModalNodePath] = useState('');
  const [suggestions, setSuggestions] = useState([]);
  const [selectedSuggestionIndex, setSelectedSuggestionIndex] = useState(-1);
  const [runError, setRunError] = useState(null);
  const [isSubmittingRun, setIsSubmittingRun] = useState(false);
  const nodePathInputRef = useRef(null);

  /* ==================================================================
     Helpers -- placeholder scanning & Monaco decorations
     ================================================================== */
     const updateLineMarkers = useCallback((content) => {
      if (!editorRef.current || !monacoRef.current) return;
      const monaco = monacoRef.current;
      const editor = editorRef.current;

      const newPlaceholderLines = [];
      const newAttentionLines = [];
      const placeholderDecorations = [];
      const attentionDecorations = [];

      content.split('\n').forEach((line, idx) => {
        const lineNo = idx + 1;
        if (PLACEHOLDER_REGEX.test(line)) {
          newPlaceholderLines.push(lineNo);
          placeholderDecorations.push({
            range: new monaco.Range(lineNo, 1, lineNo, 1),
            options: {
              isWholeLine: true,
              className: 'placeholder-line-highlight',
              hoverMessage: { value: 'Replace !!PLACEHOLDER!! before running' },
            },
          });
        } else if (ATTENTION_REGEX.test(line)) {
            newAttentionLines.push(lineNo);
            attentionDecorations.push({
              range: new monaco.Range(lineNo, 1, lineNo, 1),
              options: {
                isWholeLine: true,
                className: 'attention-line-highlight',
                hoverMessage: { value: 'Note: !!ATTENTION!! marker present' },
              },
            });
        }
      });

      setPlaceholderLines(newPlaceholderLines);
      setAttentionLines(newAttentionLines);

      // Update decorations using deltaDecorations
      placeholderDecorationsRef.current = editor.deltaDecorations(
        placeholderDecorationsRef.current,
        placeholderDecorations
      );
      attentionDecorationsRef.current = editor.deltaDecorations(
        attentionDecorationsRef.current,
        attentionDecorations
      );
    }, []); // Removed dependency on editorRef/monacoRef as they are refs

  /* ==================================================================
     Initial load
     ================================================================== */
     useEffect(() => {
      if (!configId) return;
  
      (async () => {
        try {
          setIsLoading(true);
          const text = await fetchConfigContent(configId);
          setOriginalContent(text);
          setCurrentContent(text);
          setTimeout(() => updateLineMarkers(text), 0);
        } catch (e) {
          console.error(e);
          setEditorError(e.message || 'Failed to load configuration content.');
        } finally {
          setIsLoading(false);
        }
      })();
    }, [configId]);

  /* ==================================================================
     Editor change handler
     ================================================================== */
  const handleEditorChange = (value) => {
    const newVal = value || '';
    setCurrentContent(newVal);
    if (saveSuccess) setSaveSuccess(false);
    if (editorError) setEditorError(null);
    if (runError) setRunError(null);
    updateLineMarkers(newVal);
  };

  /* ==================================================================
     Save / Reset
     ================================================================== */
  const handleSave = useCallback(async () => {
    if (!isDirty || isSaving || !configId) return;
    try {
      setIsSaving(true);
      await saveConfigContent(configId, currentContent);
      setOriginalContent(currentContent);
      setSaveSuccess(true);
      setTimeout(() => setSaveSuccess(false), 3000);
    } catch (e) {
      console.error(e);
      setEditorError(e.message || 'Failed to save configuration.');
    } finally {
      setIsSaving(false);
    }
  }, [configId, currentContent, isDirty, isSaving]);

  const handleReset = () => {
    if (!isDirty) return;
    setCurrentContent(originalContent);
    setEditorError(null);
    setSaveSuccess(false);
    setRunError(null);
    updateLineMarkers(originalContent);
  };

  /* Ctrl/Cmd-S shortcut */
  useEffect(() => {
    const keydown = (e) => {
      if ((e.metaKey || e.ctrlKey) && e.key === 's') {
        e.preventDefault();
        if (isDirty) handleSave();
      }
    };
    window.addEventListener('keydown', keydown);
    return () => window.removeEventListener('keydown', keydown);
  }, [handleSave, isDirty]);

  /* ==================================================================
     Monaco mount
     ================================================================== */
  function handleEditorDidMount(editor, monaco) {
    editorRef.current = editor;
    monacoRef.current = monaco;
    updateLineMarkers(currentContent); // Use the correct function name
  }

  /* ==================================================================
     Run button & modal logic
     ================================================================== */
  const sendRunRequest = useCallback(
    async (nodePath, parameters) => {
      if (isSubmittingRun) return;
      try {
        console.log('Sending run request:', { nodePath, parameters });
        setIsSubmittingRun(true);
        const result = await runPipeline(nodePath, parameters);
        setIsRunModalOpen(false);
        navigate(`/monitor?pipeline_id=${result.pipeline_id}`);
      } catch (e) {
        console.error('Error during run request:', e);
        setRunError(
          e.message || 'An unknown error occurred while trying to run the pipeline.'
        );
      } finally {
        setIsSubmittingRun(false);
      }
    },
    [navigate, isSubmittingRun]
  );

  const handleRunClick = async () => {
    console.log('handleRunClick invoked!');
    setRunError(null);

    /* Warn but still allow running if placeholders exist */
    if (placeholderLines.length > 0) {
      console.warn(
        'Running pipeline while !!PLACEHOLDER!! values are still present.'
      );
    }

    let parsedConfig = {};
    try {
      parsedConfig = yaml.load(currentContent) || {};
      if (typeof parsedConfig !== 'object') {
        throw new Error('Invalid YAML: Config must be an object.');
      }
    } catch (e) {
      console.error('YAML Parse Error in handleRunClick:', e);
      setRunError(`YAML Parse Error: ${e.message}`);
      return;
    }

    const nodePathFromConfig = parsedConfig.pipeline;
    if (nodePathFromConfig && typeof nodePathFromConfig === 'string') {
      if (isDirty) {
        console.warn(
          'Running pipeline with unsaved changes in the editor.'
        );
      }
      sendRunRequest(nodePathFromConfig, parsedConfig);
      return;
    }

    /* Fallback modal flow */
    console.log("Fallback: Attempting to open modal");
    setModalNodePath('');
    setSuggestions([]);
    setSelectedSuggestionIndex(-1);
    setIsRunModalOpen(true);
    console.log("Fallback: Set isRunModalOpen to true");
    if (availablePipelines.length === 0) fetchAvailablePipelinesApiCall();
    setTimeout(() => nodePathInputRef.current?.focus(), 100);
  };

  const fetchAvailablePipelinesApiCall = async () => {
    if (isFetchingPipelines) return;
    try {
      setIsFetchingPipelines(true);
      const data = await fetchAvailablePipelines();
      setAvailablePipelines(Array.isArray(data) ? data : []);
    } catch (e) {
      console.error(e);
      setRunError('Could not load pipeline suggestions.');
    } finally {
      setIsFetchingPipelines(false);
    }
  };

  // --- Modal Interaction Logic ---
  const handleNodePathChange = (e) => {
    const value = e.target.value;
    setModalNodePath(value);
    setSelectedSuggestionIndex(-1);
    if (value && availablePipelines.length > 0) {
      const lowerValue = value.toLowerCase();
      setSuggestions(availablePipelines.filter(p => p.toLowerCase().includes(lowerValue)).slice(0, 5));
    } else {
      setSuggestions([]);
    }
  };

  const handleSuggestionClick = (suggestion) => {
    setModalNodePath(suggestion);
    setSuggestions([]);
    setSelectedSuggestionIndex(-1);
    nodePathInputRef.current?.focus();
  };

  const handleModalKeyDown = (e) => {
    if (suggestions.length === 0) return;
    if (e.key === 'ArrowDown') {
      e.preventDefault();
      setSelectedSuggestionIndex(prev => (prev + 1) % suggestions.length);
    } else if (e.key === 'ArrowUp') {
      e.preventDefault();
      setSelectedSuggestionIndex(prev => (prev - 1 + suggestions.length) % suggestions.length);
    } else if (e.key === 'Enter' && selectedSuggestionIndex !== -1) {
      e.preventDefault();
      handleSuggestionClick(suggestions[selectedSuggestionIndex]);
    } else if (e.key === 'Escape') {
      setSuggestions([]);
      setSelectedSuggestionIndex(-1);
    }
  };

  const handleModalSubmit = () => {
      if (!modalNodePath.trim()) {
          setRunError("Pipeline node path cannot be empty.");
          return;
      }
       let parsedConfig = {};
      try {
         parsedConfig = yaml.load(currentContent);
         if (typeof parsedConfig !== 'object' || parsedConfig === null) throw new Error(); 
      } catch (e) {
          setRunError('Internal Error: Invalid YAML content detected before sending run request.')
          return;
      }
      sendRunRequest(modalNodePath.trim(), parsedConfig);
  };

  /* ==================================================================
     UI helpers
     ================================================================== */
  const hasPlaceholders = placeholderLines.length > 0;
  const hasAttention = attentionLines.length > 0;
  const buttonBase =
    'px-4 py-2 rounded font-semibold transition-colors disabled:opacity-50 disabled:cursor-not-allowed text-sm h-9 flex items-center justify-center';
  const resetButtonClass = `${buttonBase} ${
    isDirty
      ? 'bg-red-600 hover:bg-red-500 text-white'
      : 'bg-gray-600 text-gray-400'
  }`;
  const saveButtonClass = `${buttonBase} ${
    isDirty
      ? 'bg-blue-600 hover:bg-blue-500 text-white'
      : 'bg-gray-600 text-gray-400'
  }`;
  const backButtonClass = 'text-gray-300 hover:text-white hover:bg-gray-700';

  /* ==================================================================
     Render
     ================================================================== */
  console.log('Render - Button states:', { isSaving, isLoading, isSubmittingRun });
  const calculatedDisabled = isSaving || isLoading || isSubmittingRun;
  console.log('Render - Calculated disabled:', calculatedDisabled);
  console.log('Render - isDirty:', isDirty, 'hasPlaceholders:', hasPlaceholders);
  console.log('Render - hasAttention:', hasAttention); // Added log for attention state

  return (
    <div className='flex flex-col h-full p-4'>
      {/* ---------------- Header ---------------- */}
      <div className='flex justify-between items-center mb-4 gap-4'>
        {/* Title + Back */}
        <div className='flex items-center gap-3 min-w-0'>
          <Link
            to='/configs'
            className={`${buttonBase} ${backButtonClass} pl-2 pr-3`}
          >
            <FiChevronLeft className='w-5 h-5 mr-1' /> Back
          </Link>
          <div className='min-w-0'>
            <h1 className='text-xl lg:text-2xl font-bold text-gray-100 truncate'>
              Config Editor
            </h1>
            <p
              className='text-gray-400 text-xs lg:text-sm mt-1 truncate'
              title={configId}
            >
              Editing:{' '}
              <code className='bg-gray-700 px-1 rounded'>{configId}</code>
            </p>
          </div>
        </div>

        {/* Action buttons + placeholder/attention warning */}
        <div className='flex flex-col items-end gap-1 flex-shrink-0'>
            {/* Row for messages */}
            <div className="flex items-center justify-end gap-3 text-xs sm:text-sm h-5">
              {hasPlaceholders && (
                <span
                  className='text-amber-300 italic whitespace-nowrap'
                  title='Replace all !!PLACEHOLDER!! values before final run'
                >
                  ⚠️ Placeholders on line{placeholderLines.length > 1 ? 's' : ''}:{' '}
                  {placeholderLines.join(', ')}
                </span>
              )}
               {hasAttention && (
                <span
                  className='text-cyan-300 italic whitespace-nowrap'
                  title='Note: !!ATTENTION!! markers present on these lines'
                >
                  ℹ️ Attention on line{attentionLines.length > 1 ? 's' : ''}:{' '}
                  {attentionLines.join(', ')}
                </span>
              )}
              {saveSuccess && (
                <span className='text-green-400 italic'>Saved!</span>
              )}
            </div>

            {/* Row for buttons */}
            <div className="flex items-center gap-3">
              <EnhancedButton
                onClick={handleRunClick}
                disabled={calculatedDisabled}
                className={`${buttonBase} ${
                  !isDirty && !hasPlaceholders // Attention markers do NOT affect pulse
                    ? 'animate-pulse-green-download'
                    : 'bg-gray-600 hover:bg-gray-500 text-gray-200'
                }`}
                title={
                  !isDirty && !hasPlaceholders
                    ? 'Run with current saved configuration'
                    : hasPlaceholders
                    ? 'Run (placeholders present)'
                    : 'Run (unsaved changes present)'
                }
                particleColor="#34d399"
              >
                Run
              </EnhancedButton>
              <button
                onClick={handleReset}
                disabled={!isDirty || isSaving}
                className={resetButtonClass}
              >
                Reset
              </button>
              <button
                onClick={handleSave}
                disabled={!isDirty || isSaving}
                className={saveButtonClass}
              >
                {isSaving ? 'Saving…' : 'Save'}
              </button>
            </div>
        </div>
      </div>

      {/* ---------------- Errors ---------------- */}
      {(editorError || runError) && (
        <div className='text-red-400 bg-red-900 bg-opacity-50 p-3 rounded border border-red-700 mb-4 text-sm'>
          {editorError && <p>Editor/Save Error: {editorError}</p>}
          {runError && !isRunModalOpen && <p>Run Error: {runError}</p>}
        </div>
      )}

      {/* ---------------- Monaco Editor ---------------- */}
      <div className='flex-grow relative border border-gray-700 rounded overflow-hidden min-h-[200px]'>
        {isLoading ? (
          <div className='absolute inset-0 flex items-center justify-center bg-gray-800'>
            <FiLoader className='animate-spin text-gray-500 text-4xl' />
            <p className='text-gray-400 ml-3'>Loading configuration…</p>
          </div>
        ) : editorError && !originalContent ? (
          <div className='absolute inset-0 flex items-center justify-center bg-gray-800 text-gray-500 px-4'>
            Could not load content.
          </div>
        ) : (
          <Editor
            height='100%'
            language='yaml'
            theme='vs-dark'
            value={currentContent}
            onChange={handleEditorChange}
            onMount={handleEditorDidMount}
            options={{
              minimap: { enabled: true },
              scrollBeyondLastLine: false,
              fontSize: 14,
              wordWrap: 'on',
              automaticLayout: true,
            }}
          />
        )}
      </div>

      {/* ------------------- Modal stays unchanged ------------------- */}
      <Modal isOpen={isRunModalOpen} onClose={() => { setIsRunModalOpen(false); setRunError(null); }} title="Specify Pipeline Node Path">
            <p className="text-sm text-gray-400 mb-3">
The key <code className="bg-gray-700 px-1 rounded text-xs">pipeline:</code> was not found in your configuration.
Please provide the Python node path for the pipeline you want to run (e.g., <code className="bg-gray-700 px-1 rounded text-xs">pipelines.my_pipeline.run</code>).
            </p>
            <div className="relative">
                <input
                    ref={nodePathInputRef}
                    type="text"
                    value={modalNodePath}
                    onChange={handleNodePathChange}
                    onKeyDown={handleModalKeyDown}
                    placeholder="Enter node path (e.g., pipelines.my_pipeline.run)"
                    className="w-full px-3 py-2 bg-gray-700 border border-gray-600 rounded text-gray-100 focus:outline-none focus:border-blue-500 focus:ring-1 focus:ring-blue-500"
                    disabled={isSubmittingRun || isFetchingPipelines}
                />
                {isFetchingPipelines && <FiLoader className="animate-spin absolute right-3 top-1/2 transform -translate-y-1/2 text-gray-400"/>}
                {suggestions.length > 0 && (
                    <ul className="absolute z-10 w-full mt-1 bg-gray-700 border border-gray-600 rounded shadow-lg max-h-40 overflow-y-auto">
                        {suggestions.map((s, index) => (
                            <li
                                key={s}
                                className={`px-3 py-2 cursor-pointer hover:bg-gray-600 ${index === selectedSuggestionIndex ? 'bg-blue-700' : ''}`}
                                onClick={() => handleSuggestionClick(s)}
                            >
                                {s}
                            </li>
                        ))}
                    </ul>
                )}
            </div>

            {/* Expandable List of All Pipelines */}
            {!isFetchingPipelines && availablePipelines.length > 0 && (
                <details className="mt-2 mb-3 rounded border border-gray-600 overflow-hidden">
                    <summary className="px-3 py-2 cursor-pointer bg-gray-700 hover:bg-gray-650 text-sm text-gray-300 flex justify-between items-center">
                        <span>Show All Available Pipelines ({availablePipelines.length})</span>
                        {/* Simple triangle indicator */}
                        <span className="details-marker">▼</span>
                    </summary>
                    <ul className="bg-gray-800 max-h-48 overflow-y-auto p-2">
                        {availablePipelines
                            .slice() // Create a copy
                            .sort((a, b) => a.localeCompare(b)) // Sort alphabetically
                            .map((pipeline) => (
                            <li
                                key={pipeline}
                                className="px-3 py-1.5 cursor-pointer hover:bg-gray-700 text-gray-200 text-sm rounded"
                                onClick={(e) => {
                                    handleSuggestionClick(pipeline);
                                    // Optionally close the details element
                                    e.target.closest('details')?.removeAttribute('open');
                                }}
                            >
                                {pipeline}
                            </li>
                        ))}
                    </ul>
                </details>
            )}
             {/* Add CSS for details marker if not already global */}
             <style>{`
                details > summary { list-style: none; }
                details > summary::-webkit-details-marker { display: none; }
                details > summary .details-marker { transition: transform 0.2s; }
                details[open] > summary .details-marker { transform: rotate(180deg); }
            `}</style>

            {runError && <p className="text-red-400 text-xs mt-2">{runError}</p>}
           <div className="flex justify-end gap-3 mt-4">
               <button
                   type="button"
                   onClick={() => { setIsRunModalOpen(false); setRunError(null); }}
                   className="px-4 py-2 rounded bg-gray-600 hover:bg-gray-500 text-gray-200 transition-colors disabled:opacity-50"
                   disabled={isSubmittingRun}
               >
                   Cancel
               </button>
               <button
                   type="button"
                   onClick={handleModalSubmit}
                   className="px-4 py-2 rounded bg-blue-600 hover:bg-blue-500 text-white transition-colors disabled:opacity-50 flex items-center"
                   disabled={isSubmittingRun || !modalNodePath.trim()}
               >
                   {isSubmittingRun && <FiLoader className="animate-spin mr-2"/>} Run Pipeline
               </button>
           </div>
       </Modal>

      {/* ---------- Monaco decoration CSS ---------- */}
      <style>{`
        .placeholder-line-highlight {
          background-color: rgba(252, 211, 77, 0.25) !important; /* amber-200 */
        }
        .attention-line-highlight {
          background-color: rgba(103, 232, 249, 0.2) !important; /* cyan-300 */
          border-left: 3px solid #06b6d4; /* cyan-600 */
        }
      `}</style>
    </div>
  );
}

export default ConfigEditor;
