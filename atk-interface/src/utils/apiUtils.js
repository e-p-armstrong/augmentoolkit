const API_BASE_URL = 'http://127.0.0.1:8000'; // Ensure this matches your API server

/**
 * Constructs the full API URL.
 * @param {string} path - The endpoint path (e.g., '/pipelines/available')
 * @returns {string} The full API URL.
 */
export const getApiUrl = (path) => {
  // Ensure path starts with a single '/'
  const formattedPath = path.startsWith('/') ? path : `/${path}`;
  return `${API_BASE_URL}${formattedPath}`;
};

/**
 * Handles API errors by parsing the response and throwing a formatted error.
 * @param {Response} response - The fetch response object.
 * @param {string} [context='API call'] - A string describing the context of the API call for better error messages.
 * @throws {Error} Throws a formatted error with status and potentially parsed details.
 */
export const handleApiError = async (response, context = 'API call') => {
  let errorMsg = `${context} failed: ${response.status} ${response.statusText}`;
  let errorData = null;
  try {
    // Attempt to parse error detail from backend JSON
    errorData = await response.json();
    const detail = errorData.detail || (typeof errorData === 'string' ? errorData : null) || JSON.stringify(errorData);
     if (detail) {
        errorMsg = `${context} failed: ${detail}`;
     } 
  } catch (jsonError) {
    // If JSON parsing fails, try reading as text
    try {
      const textError = await response.text();
      if (textError) {
          errorMsg = `${context} failed: ${textError}`;
      }
    } catch (textError) {
      // Ignore text reading error, stick with the status text message
    }
  }
  console.error(`API Error during ${context}:`, { 
    status: response.status, 
    statusText: response.statusText, 
    data: errorData, 
    message: errorMsg 
  });
  const error = new Error(errorMsg);
  error.status = response.status;
  error.data = errorData; // Attach parsed data if available
  throw error;
}; 