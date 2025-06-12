import { handleApiError, getApiUrl } from './utils/apiUtils';

const API_BASE_URL = 'http://127.0.0.1:8000'; // Or use environment variable

// Helper to handle common fetch logic and errors
async function handleApiResponse(response) {
  if (!response.ok) {
    let errorMsg = `API Error: ${response.status} ${response.statusText}`;
    let errorData = null;
    try {
      // Attempt to parse error detail from backend
      errorData = await response.json();
      errorMsg = errorData.detail || JSON.stringify(errorData) || errorMsg;
    } catch (e) {
      // If parsing fails, try reading as text
      try {
          errorMsg = await response.text() || errorMsg;
      } catch (textError) { /* Ignore text reading error */ }
    }
    console.error('API Error Data:', errorData);
    const error = new Error(errorMsg);
    error.status = response.status;
    error.data = errorData; 
    throw error;
  }
  // If response has content, parse as JSON, otherwise return the response object
  const contentType = response.headers.get("content-type");
  if (contentType && contentType.indexOf("application/json") !== -1) {
    return response.json();
  }
   if (contentType && contentType.indexOf("text/") !== -1) {
      return response.text(); // Handle text responses like config content
   }
  // For downloads or other types, return the raw response
  return response;
}

// --- Config API --- 

export async function fetchConfigStructure(relativePath = '') {
  const path = relativePath ? `/${encodeURIComponent(relativePath)}` : '.';
  const response = await fetch(`${API_BASE_URL}/configs/structure/${path}`);
  return handleApiResponse(response);
}

export async function fetchConfigContent(relativePath) {
  if (!relativePath) throw new Error('Relative path is required for fetchConfigContent');
  const response = await fetch(`${API_BASE_URL}/configs/content/${encodeURIComponent(relativePath)}`);
  return handleApiResponse(response);
}

export async function saveConfigContent(relativePath, content) {
  if (!relativePath) throw new Error('Relative path is required for saveConfigContent');
  const response = await fetch(`${API_BASE_URL}/configs/content/${encodeURIComponent(relativePath)}`, {
    method: 'POST',
    headers: { 'Content-Type': 'text/plain' },
    body: content,
  });
  return handleApiResponse(response);
}

/**
 * Creates a new directory within the configs folder.
 * @param {string} relativePath - The relative path for the new directory.
 * @returns {Promise<object>} - API response.
 */
export const createConfigDirectory = async (relativePath) => {
  const url = getApiUrl('/configs/directory');
  const response = await fetch(url, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ relative_path: relativePath }),
  });
  // Use handleApiResponse which includes error handling
  return handleApiResponse(response);
};

/**
 * Deletes a specific config file.
 * @param {string} relativePath - The relative path of the file to delete.
 * @returns {Promise<object>} - API response.
 */
export const deleteConfigItem = async (relativePath) => {
    const url = getApiUrl(`/configs/${encodeURIComponent(relativePath)}`);
    const response = await fetch(url, { method: 'DELETE' });
    // Use handleApiResponse which includes error handling
    return handleApiResponse(response);
};

/**
 * Moves/Renames a file or folder within the configs directory.
 * @param {string} sourceRelativePath - Current relative path.
 * @param {string} destinationRelativePath - New relative path.
 * @returns {Promise<object>} - API response.
 */
export const moveConfigItem = async (sourceRelativePath, destinationRelativePath) => {
    const url = getApiUrl('/configs/move');
    const response = await fetch(url, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
            source_relative_path: sourceRelativePath,
            destination_relative_path: destinationRelativePath,
         }),
    });
    // Use handleApiResponse which includes error handling
    return handleApiResponse(response);
};

/**
 * Duplicates a config file from a source alias to a destination path.
 * @param {string} sourceAlias - The alias of the config file to duplicate.
 * @param {string} destinationRelativePath - The destination path for the new file.
 * @returns {Promise<object>} - API response.
 */
export const duplicateConfig = async (sourceAlias, destinationRelativePath) => {
    const url = getApiUrl('/configs/duplicate');
    const response = await fetch(url, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
            source_alias: sourceAlias,
            destination_relative_path: destinationRelativePath,
         }),
    });
    // Use handleApiResponse which includes error handling
    return handleApiResponse(response);
};

/**
 * Fetches available config aliases (e.g., from super_config.yaml) that can be used as sources for duplication.
 * NOTE: Assumes a backend endpoint GET /configs/aliases exists.
 * @returns {Promise<string[]>} - A list of config alias strings.
 */
export const fetchAvailableConfigAliases = async () => {
  // TODO: Verify/Implement the corresponding backend endpoint GET /configs/aliases
  const url = getApiUrl('/configs/aliases'); // Assuming this endpoint exists
  const response = await fetch(url);
  // Use handleApiResponse which includes error handling
  return handleApiResponse(response);
};

// --- Pipeline API --- 

export const fetchAvailablePipelines = async () => {
  const response = await fetch(getApiUrl('/pipelines/available'));
  if (!response.ok) throw new Error('Failed to fetch available pipelines');
  return response.json();
};

export async function runPipeline(nodePath, parameters) {
    const response = await fetch(`${API_BASE_URL}/pipelines/run`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ node_path: nodePath, parameters: parameters }),
      });
    return handleApiResponse(response);
}

// --- Input API --- 

export const fetchInputStructure = async (relativePath = '.') => {
    const url = getApiUrl(`/inputs/structure/${encodeURIComponent(relativePath)}`);
    const response = await fetch(url);
    if (!response.ok) {
        await handleApiError(response, 'fetch input structure');
    }
    return response.json();
};

export const uploadInputFiles = async (relativePath, files) => {
    const url = getApiUrl(`/inputs/upload/${encodeURIComponent(relativePath)}`);
    const formData = new FormData();
    files.forEach(file => formData.append('files', file, file.name)); // Use 'files' as key

    const response = await fetch(url, {
        method: 'POST',
        body: formData,
        // No 'Content-Type' header, browser sets it with boundary for multipart/form-data
    });

    if (!response.ok && response.status !== 207) { // 207 Multi-Status might indicate partial success
        await handleApiError(response, 'upload input files');
    }
    // Even on 207, parse the body for details
    const result = await response.json();
    if (response.status === 207 && result.errors?.length > 0) {
        // Optionally throw a more specific error or handle partial failures
        console.warn('Partial upload failure:', result.errors);
        // Example: throw new Error(`Upload completed with errors: ${result.errors.join(', ')}`);
    }
     if (!response.ok && response.status !== 207) {
         // Throw if it wasn't OK and wasn't a 207
         throw new Error(result.message || 'Upload failed');
     }
    return result;
};

export const downloadInputItem = async (relativePath) => {
    const url = getApiUrl(`/inputs/download/${encodeURIComponent(relativePath)}`);
    const response = await fetch(url);
    if (!response.ok) {
        await handleApiError(response, `download input item: ${relativePath}`);
    }
    // Don't parse JSON, return the raw response for blob processing
    return response;
};

export const deleteInputItem = async (relativePath) => {
    const url = getApiUrl(`/inputs/${encodeURIComponent(relativePath)}`);
    const response = await fetch(url, { method: 'DELETE' });
    if (!response.ok) {
        await handleApiError(response, `delete input item: ${relativePath}`);
    }
    return response.json(); // Or just check status if no body expected on success
};

export const createInputDirectory = async (relativePath) => {
  const url = getApiUrl('/inputs/directory');
  const response = await fetch(url, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ relative_path: relativePath }),
  });
  if (!response.ok) {
    await handleApiError(response, `create input directory: ${relativePath}`);
  }
  return response.json();
};

/**
 * Moves a file or folder within the inputs directory.
 * @param {string} sourceRelativePath - The current relative path of the item to move.
 * @param {string} destinationRelativePath - The new relative path for the item.
 * @returns {Promise<object>} - The response object from the API.
 */
export const moveInputItem = async (sourceRelativePath, destinationRelativePath) => {
    const url = getApiUrl('/inputs/move');
    const response = await fetch(url, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
            source_relative_path: sourceRelativePath,
            destination_relative_path: destinationRelativePath,
         }),
    });
    if (!response.ok) {
        await handleApiError(response, `move input item from ${sourceRelativePath} to ${destinationRelativePath}`);
    }
    return response.json();
};

// --- Output API ---

export const fetchOutputStructure = async (relativePath = '.') => {
    const url = getApiUrl(`/outputs/structure/${encodeURIComponent(relativePath)}`);
    const response = await fetch(url);
    if (!response.ok) {
        await handleApiError(response, 'fetch output structure');
    }
    return response.json();
};

export const downloadOutputItem = async (relativePath) => {
    const url = getApiUrl(`/outputs/download/${encodeURIComponent(relativePath)}`);
    const response = await fetch(url);
    if (!response.ok) {
        await handleApiError(response, `download output item: ${relativePath}`);
    }
    // Return raw response for blob processing
    return response;
};

export const deleteOutputItem = async (relativePath) => {
    const url = getApiUrl(`/outputs/${encodeURIComponent(relativePath)}`);
    const response = await fetch(url, { method: 'DELETE' });
    if (!response.ok) {
        await handleApiError(response, `delete output item: ${relativePath}`);
    }
    return response.json(); // Or check status
};

export const moveOutputItem = async (sourceRelativePath, destinationRelativePath) => {
    const url = getApiUrl('/outputs/move');
    const response = await fetch(url, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
            source_relative_path: sourceRelativePath,
            destination_relative_path: destinationRelativePath,
        }),
    });
    if (!response.ok) {
        await handleApiError(response, `move output item from ${sourceRelativePath} to ${destinationRelativePath}`);
    }
    return response.json();
};

// --- Task API ---

/**
 * Fetches the status of a specific task.
 * @param {string} taskId - The ID of the task.
 * @returns {Promise<object>} - Task status object { task_id, status, message?, progress?, details? }.
 */
export const fetchTaskStatus = async (taskId) => {
    if (!taskId) throw new Error('Task ID is required for fetchTaskStatus');
    const url = getApiUrl(`/tasks/${encodeURIComponent(taskId)}/status`);
    const response = await fetch(url);
    if (!response.ok) {
        await handleApiError(response, `fetch task status for ${taskId}`);
    }
    return response.json();
};

/**
 * Fetches the parameters used for a specific task run.
 * @param {string} taskId - The ID of the task.
 * @returns {Promise<object>} - Object containing { task_id, parameters }.
 */
export const fetchTaskParameters = async (taskId) => {
    if (!taskId) throw new Error('Task ID is required for fetchTaskParameters');
    const url = getApiUrl(`/tasks/${encodeURIComponent(taskId)}/parameters`);
    const response = await fetch(url);
    // Handle 404 specifically for parameters, as it might just mean they weren't stored/expired
    if (response.status === 404) {
        console.warn(`Parameters not found for task ${taskId}. They might not have been stored or may have expired.`);
        return { task_id: taskId, parameters: null }; // Return null for parameters
    }
    if (!response.ok) {
        await handleApiError(response, `fetch task parameters for ${taskId}`);
    }
    return response.json();
};

/**
 * Interrupts a running task or revokes a pending one.
 * @param {string} taskId - The ID of the task.
 * @returns {Promise<object>} - API response message.
 */
export const interruptTask = async (taskId) => {
    if (!taskId) throw new Error('Task ID is required for interruptTask');
    const url = getApiUrl(`/tasks/${encodeURIComponent(taskId)}/interrupt`);
    const response = await fetch(url, { method: 'POST' });
    if (!response.ok) {
        await handleApiError(response, `interrupt task ${taskId}`);
    }
    return response.json();
};

/**
 * Fetches the list of pending and scheduled tasks.
 * @returns {Promise<object>} - Object containing { pending_tasks, scheduled_tasks, message }.
 */
export const fetchTaskQueue = async () => {
    const url = getApiUrl('/tasks/queue');
    const response = await fetch(url);
    if (!response.ok) {
        await handleApiError(response, 'fetch task queue');
    }
    return response.json();
};

/**
 * Fetches the logs for a specific task.
 * @param {string} taskId - The ID of the task.
 * @param {number} [tail] - Optional number of lines to tail.
 * @returns {Promise<object>} - Object containing { task_id, message, logs }.
 */
export const fetchTaskLogs = async (taskId, tail) => {
    if (!taskId) throw new Error('Task ID is required for fetchTaskLogs');
    let url = getApiUrl(`/tasks/${encodeURIComponent(taskId)}/logs`);
    if (tail !== undefined && tail !== null) {
        url += `?tail=${tail}`;
    }
    const response = await fetch(url);
    // Handle 404 gracefully, log file might not exist yet
    if (response.status === 404) {
         console.log(`Log file for task ${taskId} not found.`);
         return { task_id: taskId, message: "Log file not found.", logs: "" };
    }
    if (!response.ok) {
        await handleApiError(response, `fetch task logs for ${taskId}`);
    }
    return response.json();
};

/**
 * Deletes the log file for a specific task.
 * @param {string} taskId - The ID of the task.
 * @returns {Promise<object>} - API response message.
 */
export const deleteTaskLogs = async (taskId) => {
    if (!taskId) throw new Error('Task ID is required for deleteTaskLogs');
    const url = getApiUrl(`/tasks/${encodeURIComponent(taskId)}/logs`);
    const response = await fetch(url, { method: 'DELETE' });
    if (!response.ok) {
        await handleApiError(response, `delete task logs for ${taskId}`);
    }
    return response.json();
};

/**
 * Downloads the output directory for a specific task as a zip file.
 * @param {string} taskId - The ID of the task.
 * @returns {Promise<Response>} - The raw fetch Response object for blob processing.
 */
export const downloadTaskOutput = async (taskId) => {
    if (!taskId) throw new Error('Task ID is required for downloadTaskOutput');
    const url = getApiUrl(`/tasks/${encodeURIComponent(taskId)}/outputs/download`);
    const response = await fetch(url);
    if (!response.ok) {
        // Handle 404 specifically - output might not exist
         if (response.status === 404) {
             throw new Error(`Output directory not found for task ${taskId}.`);
         }
        await handleApiError(response, `download task output for ${taskId}`);
    }
    // Return raw response for blob processing
    return response;
};

/**
 * Fetches the list of available log files.
 * GET /logs
 * @returns {Promise<string[]>} A promise that resolves to an array of log filenames.
 */
export const fetchLogFiles = async () => {
    const response = await fetchWithErrorHandling(`${API_BASE_URL}/logs`);
    const data = await response.json();
    // Ensure we return an array, even if the API response structure is slightly off
    return Array.isArray(data?.log_files) ? data.log_files : [];
};

/**
 * Deletes a specific log file based on its task ID.
 * DELETE /tasks/{task_id}/logs
 * @param {string} taskId - The task ID (extracted from the log filename).
 * @returns {Promise<object>} A promise that resolves to the API response message.
 */
export const deleteLogFile = async (taskId) => {
    if (!taskId) {
        throw new Error("Task ID is required to delete a log file.");
    }
    // Basic check to prevent deleting files without UUID-like format, adjust if task IDs differ
    // if (!/^[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}$/.test(taskId)) {
    //     console.warn(`Attempting to delete log for potentially invalid Task ID: ${taskId}`);
    // }
    const response = await fetchWithErrorHandling(`${API_BASE_URL}/tasks/${encodeURIComponent(taskId)}/logs`, {
        method: 'DELETE',
    });
    return response.json();
};

/**
 * Clears all log files in the logs directory.
 * DELETE /logs
 * @returns {Promise<object>} A promise that resolves to the API response (message and potential errors).
 */
export const clearAllLogs = async () => {
    const response = await fetchWithErrorHandling(`${API_BASE_URL}/logs`, {
        method: 'DELETE',
    });
    return response.json(); // Example: {"message": "Successfully deleted X log file(s)...", "errors": []}
};

// Utility function assumed to exist from other files (InputManager/ConfigSelector context)
// If it doesn't exist, it needs to be created.
async function fetchWithErrorHandling(url, options = {}) {
    const response = await fetch(url, options);
    if (!response.ok) {
        let errorPayload;
        try {
            errorPayload = await response.json(); // Try to parse JSON error detail
        } catch (e) {
            // Ignore parsing error, use status text
        }
        const errorMessage = errorPayload?.detail || response.statusText || `Request failed with status ${response.status}`;
        console.error(`API Error (${response.status}) for ${options.method || 'GET'} ${url}:`, errorMessage, errorPayload);
        // Throw an error that includes the status code and message
        const error = new Error(errorMessage);
        error.status = response.status;
        error.payload = errorPayload;
        throw error;
    }
    return response;
}

// Utility to get the base API URL (if needed elsewhere, or for env vars)
// export const getBaseApiUrl = () => API_BASE_URL; 