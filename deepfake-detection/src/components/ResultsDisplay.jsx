import React from 'react';

const ResultsDisplay = ({ results }) => {
  if (!results) {
    return (
      <div className="my-6 rounded-xl shadow-lg overflow-hidden border border-gray-200">
        <div className="px-6 py-4 bg-gray-100 text-gray-700">
          <div className="flex justify-between items-center">
            <h2 className="text-xl font-bold">No Results Available</h2>
          </div>
        </div>
        <div className="bg-white p-6">
          <p className="text-gray-700">
            No analysis results are available. Please ensure the model processed the image successfully.
          </p>
        </div>
      </div>
    );
  }
  const isReal = results.classification === 'Real';
  const cardBorderClass = isReal ? 'border-green-200' : 'border-red-200';
  const cardBgClass = isReal ? 'bg-green-50' : 'bg-red-50';
  const headerBgClass = isReal ? 'bg-green-600' : 'bg-red-600';
  const badgeBgClass = isReal ? 'bg-green-800' : 'bg-red-800';
  const confidenceBarColor = isReal ? 'bg-green-600' : 'bg-red-600';
  const textHintClass = isReal ? 'text-green-600' : 'text-red-600';
  const buttonBgClass = isReal ? 'bg-green-600 hover:bg-green-700' : 'bg-red-600 hover:bg-red-700';
  const buttonRingClass = isReal ? 'focus:ring-green-400' : 'focus:ring-red-400';
  let confidence = parseFloat(results.confidence);
  if (isNaN(confidence) || confidence < 0) confidence = 0;
  if (confidence > 1) confidence = confidence / 100; // Handle if confidence comes as percentage value
  const confidencePct = (confidence * 100).toFixed(1);

  return (
    <div className={`my-6 rounded-xl shadow-lg overflow-hidden border ${cardBorderClass}`}>
      {/* Header section */}
      <div className={`px-6 py-4 ${headerBgClass} text-white`}>
        <div className="flex justify-between items-center">
          <h2 className="text-xl font-bold">Detection Results</h2>
          <span className={`px-3 py-1 rounded-full ${badgeBgClass} text-white text-sm font-semibold`}>
            {results.classification}
          </span>
        </div>
      </div>
      
      {/* Content section */}
      <div className="bg-white p-6">
        {/* Confidence Bar */}
        <div className="mb-6">
          <div className="flex justify-between items-center mb-2">
            <span className="font-medium text-gray-700">Confidence Level</span>
            <span className="text-sm font-bold">{confidencePct}%</span>
          </div>
          <div className="w-full bg-gray-200 rounded-full h-3">
            <div 
              className={`${confidenceBarColor} rounded-full h-3 transition-all duration-500 ease-out`}
              style={{ width: `${confidencePct}%` }}
            ></div>
          </div>
          <p className={`text-xs mt-1 ${textHintClass} italic`}>
            {isReal ? 'Higher values indicate greater confidence in real image detection' : 
              'Higher values indicate greater confidence in deepfake detection'}
          </p>
        </div>

        {/* Classification & Explanation */}
        <div className="mb-4">
          <h3 className="text-lg font-semibold mb-2 text-gray-800">Analysis Summary</h3>
          <p className="text-gray-700 leading-relaxed border-l-4 border-gray-200 pl-3 py-1">
            {results.explanation || 'No detailed explanation available for this result.'}
          </p>
        </div>
        
        {/* Features Analysis */}
        {results.features && (
          <div className="mt-6">
            <h3 className="text-lg font-semibold mb-3 text-gray-800">Key Indicators</h3>
            <div className="space-y-3">
              {results.features.map((feature, index) => (
                <div key={index} className="bg-gray-50 rounded-lg p-3">
                  <div className="flex justify-between items-center mb-1">
                    <span className="text-sm font-medium text-gray-700">{feature.name}</span>
                    <span className="text-sm text-gray-500">{(feature.value * 100).toFixed(0)}%</span>
                  </div>
                  <div className="w-full bg-gray-200 rounded-full h-2">
                    <div 
                      className={`${confidenceBarColor} rounded-full h-2`}
                      style={{ width: `${feature.value * 100}%` }}
                    ></div>
                  </div>
                </div>
              ))}
            </div>
          </div>
        )}

        {/* Action button */}
        <div className="mt-6 text-center">
          <button className={`${buttonBgClass} text-white font-medium py-2 px-6 rounded-lg shadow transition-colors duration-200 focus:outline-none focus:ring-2 ${buttonRingClass} focus:ring-opacity-50`}>
            {isReal ? 'Verify Details' : 'View Full Report'}
          </button>
        </div>
      </div>
    </div>
  );
};

export default ResultsDisplay;
