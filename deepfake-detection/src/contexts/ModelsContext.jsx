import React, { createContext, useContext, useState } from 'react';
const ModelsContext = createContext();
export function useModels() {
  return useContext(ModelsContext);
}
export default function ModelsProvider({ children, models }) {
  const [selectedModels, setSelectedModels] = useState([]);
  const [availableModels, setAvailableModels] = useState(models || []);
  const updateAvailableModels = (newModels) => {
    const efficientNetModel = newModels.find(model => 
      model.id.toLowerCase().includes('efficientnet') ||
      model.name.toLowerCase().includes('efficientnet')
    );
    
    if (efficientNetModel) {
      setAvailableModels([efficientNetModel]);
      setSelectedModels([efficientNetModel.id]);
    } else if (newModels.length > 0) {
      setAvailableModels([newModels[0]]);
      setSelectedModels([newModels[0].id]);
    } else {
      setAvailableModels([]);
      setSelectedModels([]);
    }
  };
  const value = {
    availableModels,
    selectedModels,
    comparisonMode: false, // Always disable comparison mode
    addModel: () => {}, // No-op
    removeModel: () => {}, // No-op
    toggleModel: () => {}, // No-op
    selectSingleModel: (modelId) => {
      if (availableModels.some(model => model.id === modelId)) {
        setSelectedModels([modelId]);
      }
    },
    selectAllModels: () => {
      if (availableModels.length > 0) {
        setSelectedModels([availableModels[0].id]);
      }
    },
    toggleComparisonMode: () => {}, // No-op - comparison mode is always disabled
    updateAvailableModels,
    hasMultipleModelsSelected: false // Always false since we only allow one model
  };
  
  return (
    <ModelsContext.Provider value={value}>
      {children}
    </ModelsContext.Provider>
  );
}