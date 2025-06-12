import React, { createContext, useState, useContext, useCallback, useEffect } from 'react';

const TaskHistoryContext = createContext();

export const useTaskHistory = () => useContext(TaskHistoryContext);

export const TaskHistoryProvider = ({ children }) => {
    // Initialize state from localStorage if available, otherwise empty array
    const [taskHistory, setTaskHistory] = useState(() => {
        try {
            const storedHistory = localStorage.getItem('atk_taskHistory');
            return storedHistory ? JSON.parse(storedHistory) : [];
        } catch (error) {
            console.error("Error reading task history from localStorage:", error);
            return [];
        }
    });

    // Persist history to localStorage whenever it changes
    useEffect(() => {
        try {
            localStorage.setItem('atk_taskHistory', JSON.stringify(taskHistory));
        } catch (error) {
            console.error("Error saving task history to localStorage:", error);
        }
    }, [taskHistory]);

    const addTaskToHistory = useCallback((taskId) => {
        if (!taskId) return;
        console.log(`[TaskHistoryContext] addTaskToHistory called with taskId: ${taskId}`);
        setTaskHistory(prevHistory => {
            console.log(`[TaskHistoryContext] Previous history:`, prevHistory);
            // Avoid duplicates and add to the beginning (most recent first)
            const newHistory = [taskId, ...prevHistory.filter(id => id !== taskId)];
            // Optional: Limit history size
            // const MAX_HISTORY_SIZE = 20;
            // return newHistory.slice(0, MAX_HISTORY_SIZE);
            console.log(`[TaskHistoryContext] New history:`, newHistory);
            return newHistory;
        });
    }, []);

    const clearTaskHistory = useCallback(() => {
        console.log(`[TaskHistoryContext] clearTaskHistory called.`);
        setTaskHistory([]);
        localStorage.removeItem('atk_taskHistory'); // Clear from storage too
    }, []);

    const value = {
        taskHistory,
        addTaskToHistory,
        clearTaskHistory,
    };

    return (
        <TaskHistoryContext.Provider value={value}>
            {children}
        </TaskHistoryContext.Provider>
    );
}; 