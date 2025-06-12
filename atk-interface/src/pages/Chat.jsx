import React, { useState, useEffect, useRef } from 'react';
import { FiSend, FiEdit2, FiCopy, FiCheck, FiTrash2, FiSettings, FiRefreshCw } from 'react-icons/fi';

function Chat() {
  const [messages, setMessages] = useState([]);
  const [inputMessage, setInputMessage] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [serverPort, setServerPort] = useState('8003');
  const [showPortSettings, setShowPortSettings] = useState(false);
  const [serverStatus, setServerStatus] = useState('checking'); // 'checking', 'online', 'offline'
  const [serverType, setServerType] = useState('unknown'); // 'openai', 'rag', 'unknown'
  const [editingMessageId, setEditingMessageId] = useState(null);
  const [copiedMessageId, setCopiedMessageId] = useState(null);
  const [isRefreshingStatus, setIsRefreshingStatus] = useState(false);
  const [isStreaming, setIsStreaming] = useState(true); // New streaming toggle, default true
  const messagesEndRef = useRef(null);
  const inputRef = useRef(null);

  // Check server status
  useEffect(() => {
    const checkServerStatus = async () => {
      try {
        // Use the health endpoint instead of generate with empty messages
        const response = await fetch(`http://127.0.0.1:${serverPort}/health`, {
          method: 'GET',
          headers: { 'Content-Type': 'application/json' }
        });
        
        if (response.ok) {
          const data = await response.json();
          setServerStatus('online');
          
          // Detect server type based on response message
          if (data.message && data.message.includes('RAG')) {
            setServerType('rag');
          } else if (data.message && data.message.includes('OpenAI')) {
            setServerType('openai');
          } else {
            setServerType('unknown');
          }
        } else {
          setServerStatus('offline');
          setServerType('unknown');
        }
      } catch (error) {
        setServerStatus('offline');
        setServerType('unknown');
      }
    };

    // Check immediately when port changes
    checkServerStatus();
    
    // Set up interval for periodic checks
    const interval = setInterval(checkServerStatus, 3000); // Check every 3 seconds
    return () => clearInterval(interval);
  }, [serverPort]); // Re-run when serverPort changes

  // Auto-scroll to bottom when new messages arrive
  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages]);

  const handleSendMessage = async () => {
    if (!inputMessage.trim() || isLoading || serverStatus !== 'online') return;

    const userMessage = { role: 'user', content: inputMessage.trim(), id: Date.now() };
    setMessages(prev => [...prev, userMessage]);
    setInputMessage('');
    setIsLoading(true);

    try {
      // Prepare messages for API (remove id field)
      const apiMessages = messages.concat(userMessage).map(({ role, content }) => ({ role, content }));
      
      if (isStreaming) {
        // Handle streaming response
        const assistantMessageId = Date.now() + 1;
        const assistantMessage = {
          role: 'assistant',
          content: '',
          id: assistantMessageId
        };
        
        // Add empty assistant message that we'll update
        setMessages(prev => [...prev, assistantMessage]);
        
        const response = await fetch(`http://127.0.0.1:${serverPort}/generate-stream`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ messages: apiMessages })
        });

        if (!response.ok) {
          let errorText = `Server responded with ${response.status}`;
          try {
            const errorData = await response.text();
            if (errorData) {
              errorText += `: ${errorData}`;
            }
          } catch (e) {
            // Ignore if we can't read the error body
          }
          throw new Error(errorText);
        }

        const reader = response.body.getReader();
        const decoder = new TextDecoder();
        let accumulatedContent = '';

        try {
          while (true) {
            const { done, value } = await reader.read();
            if (done) break;

            const chunk = decoder.decode(value, { stream: true });
            const lines = chunk.split('\n');

            for (const line of lines) {
              if (line.startsWith('data: ')) {
                try {
                  const data = JSON.parse(line.slice(6));
                  if (data.error) {
                    throw new Error(data.error);
                  }
                  if (data.text) {
                    accumulatedContent += data.text;
                    // Update the assistant message with accumulated content
                    setMessages(prev => prev.map(msg => 
                      msg.id === assistantMessageId 
                        ? { ...msg, content: accumulatedContent }
                        : msg
                    ));
                  }
                  if (data.done) {
                    break;
                  }
                } catch (parseError) {
                  console.warn('Failed to parse SSE data:', line);
                }
              }
            }
          }
        } finally {
          reader.releaseLock();
        }
      } else {
        // Handle non-streaming response (original logic)
        const response = await fetch(`http://127.0.0.1:${serverPort}/generate`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ messages: apiMessages })
        });

        if (!response.ok) {
          let errorText = `Server responded with ${response.status}`;
          try {
            const errorData = await response.text();
            if (errorData) {
              errorText += `: ${errorData}`;
            }
          } catch (e) {
            // Ignore if we can't read the error body
          }
          throw new Error(errorText);
        }

        const responseText = await response.text();
        
        // Try to parse as JSON first, in case the server wraps the response in quotes
        let finalResponseText;
        try {
          // If the response is JSON-wrapped (e.g., "\"Hello world\""), parse it
          const parsed = JSON.parse(responseText);
          finalResponseText = typeof parsed === 'string' ? parsed : responseText;
        } catch (e) {
          // If it's not JSON, use as-is
          finalResponseText = responseText;
        }
        
        const assistantMessage = {
          role: 'assistant',
          content: finalResponseText,
          id: Date.now() + 1
        };
        
        setMessages(prev => [...prev, assistantMessage]);
      }
    } catch (error) {
      console.error('Error sending message:', error);
      const errorMessage = {
        role: 'assistant',
        content: `Error: ${error.message}. Please check if the server is running on port ${serverPort} and try refreshing the connection.`,
        id: Date.now() + 1,
        isError: true
      };
      setMessages(prev => [...prev, errorMessage]);
      
      // Also refresh server status to check if server went offline
      handleManualRefresh();
    } finally {
      setIsLoading(false);
      inputRef.current?.focus();
    }
  };

  const handleKeyPress = (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSendMessage();
    }
  };

  const handleCopyMessage = (message) => {
    navigator.clipboard.writeText(message.content);
    setCopiedMessageId(message.id);
    setTimeout(() => setCopiedMessageId(null), 2000);
  };

  const handleEditMessage = (message) => {
    setEditingMessageId(message.id);
  };

  const handleSaveEdit = (messageId, newContent) => {
    setMessages(prev => prev.map(msg => 
      msg.id === messageId ? { ...msg, content: newContent } : msg
    ));
    setEditingMessageId(null);
  };

  const handleDeleteMessage = (messageId) => {
    setMessages(prev => prev.filter(msg => msg.id !== messageId));
  };

  const handleClearChat = () => {
    if (window.confirm('Are you sure you want to clear all messages?')) {
      setMessages([]);
    }
  };

  // Manual refresh function
  const handleManualRefresh = async () => {
    setIsRefreshingStatus(true);
    try {
      const response = await fetch(`http://127.0.0.1:${serverPort}/health`, {
        method: 'GET',
        headers: { 'Content-Type': 'application/json' }
      });
      
      if (response.ok) {
        const data = await response.json();
        setServerStatus('online');
        
        if (data.message && data.message.includes('RAG')) {
          setServerType('rag');
        } else if (data.message && data.message.includes('OpenAI')) {
          setServerType('openai');
        } else {
          setServerType('unknown');
        }
      } else {
        setServerStatus('offline');
        setServerType('unknown');
      }
    } catch (error) {
      setServerStatus('offline');
      setServerType('unknown');
    } finally {
      setIsRefreshingStatus(false);
    }
  };

  const MessageComponent = ({ message }) => {
    const [editContent, setEditContent] = useState(message.content);
    const isEditing = editingMessageId === message.id;
    const isCopied = copiedMessageId === message.id;

    return (
      <div className={`flex ${message.role === 'user' ? 'justify-end' : 'justify-start'} mb-4`}>
        <div className={`max-w-[70%] rounded-lg p-4 ${
          message.role === 'user' 
            ? 'bg-blue-600 text-white' 
            : message.isError 
              ? 'bg-red-800 text-red-100'
              : 'bg-gray-700 text-gray-100'
        }`}>
          <div className="flex items-start justify-between gap-2 mb-1">
            <span className="text-xs opacity-70 font-semibold">
              {message.role === 'user' ? 'You' : 'Assistant'}
            </span>
            <div className="flex gap-1">
              <button
                onClick={() => handleCopyMessage(message)}
                className="p-1 hover:bg-white/10 rounded transition-colors"
                title="Copy message"
              >
                {isCopied ? <FiCheck size={14} /> : <FiCopy size={14} />}
              </button>
              <button
                onClick={() => handleEditMessage(message)}
                className="p-1 hover:bg-white/10 rounded transition-colors"
                title="Edit message"
              >
                <FiEdit2 size={14} />
              </button>
              <button
                onClick={() => handleDeleteMessage(message.id)}
                className="p-1 hover:bg-white/10 rounded transition-colors"
                title="Delete message"
              >
                <FiTrash2 size={14} />
              </button>
            </div>
          </div>
          
          {isEditing ? (
            <div className="mt-2">
              <textarea
                value={editContent}
                onChange={(e) => setEditContent(e.target.value)}
                className="w-full p-2 bg-black/20 rounded text-sm resize-none"
                rows={4}
                autoFocus
              />
              <div className="flex gap-2 mt-2">
                <button
                  onClick={() => handleSaveEdit(message.id, editContent)}
                  className="px-3 py-1 bg-green-600 hover:bg-green-700 rounded text-xs"
                >
                  Save
                </button>
                <button
                  onClick={() => {
                    setEditingMessageId(null);
                    setEditContent(message.content);
                  }}
                  className="px-3 py-1 bg-gray-600 hover:bg-gray-700 rounded text-xs"
                >
                  Cancel
                </button>
              </div>
            </div>
          ) : (
            <div className="whitespace-pre-wrap break-words text-sm">
              {message.content}
            </div>
          )}
        </div>
      </div>
    );
  };

  return (
    <div className="flex flex-col h-full bg-gray-900">
      {/* Header */}
      <div className="bg-gray-800 p-4 border-b border-gray-700">
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-4">
            <h1 className="text-xl font-bold text-white">Chat Interface</h1>
            <div className="flex items-center gap-2">
              <div className={`w-2 h-2 rounded-full ${
                serverStatus === 'online' ? 'bg-green-500' : 
                serverStatus === 'offline' ? 'bg-red-500' : 
                'bg-yellow-500'
              }`} />
              <span className="text-sm text-gray-400">
                {serverStatus === 'online' 
                  ? `Server Online (${serverType === 'rag' ? 'RAG-Enabled' : serverType === 'openai' ? 'Standard' : 'Unknown Type'})` 
                  : serverStatus === 'offline' 
                    ? 'Server Offline' 
                    : 'Checking...'}
                {isRefreshingStatus && serverStatus === 'online' && (
                  <span className="ml-1 text-xs opacity-60">(refreshing)</span>
                )}
              </span>
              {serverStatus === 'online' && (
                <span className="text-xs text-gray-500 bg-gray-700 px-2 py-1 rounded">
                  Port {serverPort}
                </span>
              )}
            </div>
          </div>
          
          <div className="flex items-center gap-2">
            <button
              onClick={handleManualRefresh}
              disabled={isRefreshingStatus}
              className="p-2 hover:bg-gray-700 rounded transition-colors"
              title="Refresh server status"
            >
              <FiRefreshCw className={isRefreshingStatus ? 'animate-spin' : ''} />
            </button>
            <button
              onClick={() => setShowPortSettings(!showPortSettings)}
              className="p-2 hover:bg-gray-700 rounded transition-colors"
              title="Server settings"
            >
              <FiSettings />
            </button>
            <button
              onClick={handleClearChat}
              className="px-3 py-1 bg-red-600 hover:bg-red-700 rounded text-sm"
              disabled={messages.length === 0}
            >
              Clear Chat
            </button>
          </div>
        </div>
        
        {/* Port Settings */}
        {showPortSettings && (
          <div className="mt-4 pt-4 border-t border-gray-700">
            <div className="flex items-center gap-4 mb-3">
              <div className="flex items-center gap-2">
                <label className="text-sm text-gray-400">Server Port:</label>
                <input
                  type="text"
                  value={serverPort}
                  onChange={(e) => setServerPort(e.target.value)}
                  className="px-2 py-1 bg-gray-700 rounded text-sm w-20"
                  placeholder="8003"
                />
                <span className="text-xs text-gray-500">
                  Default: 8003 (Change if server is running on a different port)
                </span>
              </div>
            </div>
            <div className="flex items-center gap-2">
              <label className="text-sm text-gray-400">Streaming:</label>
              <button
                onClick={() => setIsStreaming(!isStreaming)}
                className={`relative inline-flex h-6 w-11 items-center rounded-full transition-colors ${
                  isStreaming ? 'bg-blue-600' : 'bg-gray-600'
                }`}
              >
                <span
                  className={`inline-block h-4 w-4 transform rounded-full bg-white transition-transform ${
                    isStreaming ? 'translate-x-6' : 'translate-x-1'
                  }`}
                />
              </button>
              <span className="text-xs text-gray-500">
                {isStreaming ? 'On' : 'Off'} - Stream responses in real-time
              </span>
            </div>
          </div>
        )}
      </div>

      {/* Messages Area */}
      <div className="flex-grow overflow-y-auto p-4">
        {messages.length === 0 ? (
          <div className="flex items-center justify-center h-full">
            <div className="text-center text-gray-500 max-w-2xl px-4">
              {serverStatus === 'online' ? (
                <>
                  <p className="text-lg mb-2">No messages yet</p>
                  <p className="text-sm">
                    Start a conversation by typing a message below
                  </p>
                </>
              ) : (
                <>
                  <p className="text-lg mb-4 text-gray-400">Welcome to the Chat Interface</p>
                  <div className="text-sm leading-relaxed space-y-3">
                    <p>
                      This page is where you can interact with models that you train using Augmentoolkit. 
                      Start an LLM server with one of the utility pipelines (<span className="font-mono bg-gray-800 px-2 py-1 rounded">basic-server</span> or <span className="font-mono bg-gray-800 px-2 py-1 rounded">rag-server</span>, in <span className="font-mono bg-gray-800 px-2 py-1 rounded">generation/utilities</span>) 
                      and when it reaches 100% completion, this page will activate and let you chat with your model.
                    </p>
                    <p className="text-xs text-gray-600">
                      <strong>Pro tip:</strong> if you want to use your custom model in production, the code for these pipelines also provides a great reference. Check it out!
                    </p>
                  </div>
                </>
              )}
            </div>
          </div>
        ) : (
          <div>
            {messages.map(message => (
              <MessageComponent key={message.id} message={message} />
            ))}
          </div>
        )}
        {isLoading && (
          <div className="flex justify-start mb-4">
            <div className="bg-gray-700 rounded-lg p-4 max-w-[70%]">
              <div className="flex items-center gap-2">
                <div className="animate-pulse flex gap-1">
                  <div className="w-2 h-2 bg-gray-400 rounded-full animate-bounce" style={{ animationDelay: '0ms' }} />
                  <div className="w-2 h-2 bg-gray-400 rounded-full animate-bounce" style={{ animationDelay: '150ms' }} />
                  <div className="w-2 h-2 bg-gray-400 rounded-full animate-bounce" style={{ animationDelay: '300ms' }} />
                </div>
                <span className="text-sm text-gray-400">Assistant is typing...</span>
              </div>
            </div>
          </div>
        )}
        <div ref={messagesEndRef} />
      </div>

      {/* Input Area */}
      <div className="border-t border-gray-700 p-4 bg-gray-800">
        <div className="flex gap-2">
          <textarea
            ref={inputRef}
            value={inputMessage}
            onChange={(e) => setInputMessage(e.target.value)}
            onKeyPress={handleKeyPress}
            placeholder={
              serverStatus === 'online' 
                ? "Type your message here... (Press Enter to send, Shift+Enter for new line)"
                : "Waiting for server connection..."
            }
            className="flex-grow p-3 bg-gray-700 text-white rounded-lg resize-none focus:outline-none focus:ring-2 focus:ring-blue-500"
            rows={3}
            disabled={serverStatus !== 'online'}
          />
          <button
            onClick={handleSendMessage}
            disabled={!inputMessage.trim() || isLoading || serverStatus !== 'online'}
            className={`px-6 py-3 rounded-lg font-semibold transition-colors flex items-center gap-2 ${
              !inputMessage.trim() || isLoading || serverStatus !== 'online'
                ? 'bg-gray-600 text-gray-400 cursor-not-allowed'
                : 'bg-blue-600 text-white hover:bg-blue-700'
            }`}
          >
            <FiSend />
            Send
          </button>
        </div>
      </div>
    </div>
  );
}

export default Chat; 