"use client";

import React, { useState } from "react";

interface PatientData {
  patientId: string;
  birthYear: number;
  gender: "male" | "female";
  hypertension: boolean;
  diabetes: boolean;
  strokeHistory: boolean;
  heartAttack: boolean;
  bloodThinners: boolean;
  initialVisualAcuity: number;
  initialThickness: number;
}

interface Prediction {
  success: boolean;
  prediction: {
    recommendation: string;
    predictedChange: "improve" | "decline" | "stable";
    confidence: number;
    nextFollowUp: string;
    currentVisualAcuity: number;
    predictedVisualAcuity: number;
  };
  charts?: {
    main_chart?: string;
  };
  statistics: {
    initialVisualAcuity: number;
    finalVisualAcuity: number;
    totalInjections: number;
  };
}

export default function App() {
  const [patientData, setPatientData] = useState<PatientData>({
    patientId: "",
    birthYear: 1948,
    gender: "male",
    hypertension: true,
    diabetes: false,
    strokeHistory: false,
    heartAttack: false,
    bloodThinners: true,
    initialVisualAcuity: 0.3,
    initialThickness: 420
  });
  const [prediction, setPrediction] = useState<Prediction | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  interface InputChangeEvent
    extends React.ChangeEvent<HTMLInputElement | HTMLSelectElement> {}

  const handleInputChange = (e: InputChangeEvent) => {
    const target = e.target as HTMLInputElement | HTMLSelectElement;
    const { name, value, type } = target;
    setPatientData(prev => ({
      ...prev,
      [name]: type === "checkbox" ? (target as HTMLInputElement).checked : value
    }));
  };

  const handlePredict = async () => {
    setLoading(true);
    setError(null);
    try {
      const response = await fetch("http://localhost:5000/api/predict", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ patientInfo: patientData })
      });
      const data = await response.json();
      if (data.success) setPrediction(data);
      else setError(data.error);
    } catch (err) {
      setError("Failed to connect to server");
    } finally {
      setLoading(false);
    }
  };

  const loadSampleData = async () => {
    try {
      const response = await fetch("http://localhost:5000/api/sample-patient");
      const data = await response.json();
      setPatientData(data.patientInfo);
    } catch (err) {
      setError("Failed to load sample data");
    }
  };

  return (
    <div className="min-h-screen bg-gray-50 py-10 px-4">
      <header className="max-w-7xl mx-auto mb-8 text-center">
        <h1 className="text-3xl font-bold text-gray-800">
          AMD Treatment Prediction System
        </h1>
      </header>

      <div className="max-w-7xl mx-auto grid grid-cols-1 lg:grid-cols-2 gap-8">
        {/* Left Column: Patient Form */}
        <div className="bg-white shadow-md rounded-xl p-6 space-y-6">
          <div className="flex justify-between items-center mb-4">
            <h2 className="text-xl font-semibold text-gray-700">
              Patient Information
            </h2>
            <button
              onClick={loadSampleData}
              className="text-sm bg-blue-500 text-white px-3 py-1 rounded hover:bg-blue-600 transition"
            >
              Load Sample Patient
            </button>
          </div>

          <div className="space-y-4">
            <div>
              <label className="block text-gray-600 mb-1">Patient ID:</label>
              <input
                type="text"
                name="patientId"
                value={patientData.patientId}
                onChange={handleInputChange}
                className="w-full border rounded-md px-3 py-2 focus:outline-none focus:ring-2 focus:ring-blue-400"
              />
            </div>

            <div>
              <label className="block text-gray-600 mb-1">Birth Year:</label>
              <input
                type="number"
                name="birthYear"
                value={patientData.birthYear}
                onChange={handleInputChange}
                className="w-full border rounded-md px-3 py-2 focus:outline-none focus:ring-2 focus:ring-blue-400"
              />
            </div>

            <div>
              <label className="block text-gray-600 mb-1">Gender:</label>
              <select
                name="gender"
                value={patientData.gender}
                onChange={handleInputChange}
                className="w-full border rounded-md px-3 py-2 focus:outline-none focus:ring-2 focus:ring-blue-400"
              >
                <option value="male">Male</option>
                <option value="female">Female</option>
              </select>
            </div>

            <div>
              <label className="block text-gray-600 mb-1">
                Initial Visual Acuity:
              </label>
              <input
                type="number"
                step="0.01"
                name="initialVisualAcuity"
                value={patientData.initialVisualAcuity}
                onChange={handleInputChange}
                className="w-full border rounded-md px-3 py-2 focus:outline-none focus:ring-2 focus:ring-blue-400"
              />
            </div>

            <div>
              <label className="block text-gray-600 mb-1">
                Initial Retinal Thickness (μm):
              </label>
              <input
                type="number"
                name="initialThickness"
                value={patientData.initialThickness}
                onChange={handleInputChange}
                className="w-full border rounded-md px-3 py-2 focus:outline-none focus:ring-2 focus:ring-blue-400"
              />
            </div>

            <div className="flex flex-wrap gap-4 mt-2">
              {[
                "hypertension",
                "diabetes",
                "strokeHistory",
                "heartAttack",
                "bloodThinners"
              ].map(key => (
                <label key={key} className="flex items-center space-x-2">
                  <input
                    type="checkbox"
                    name={key}
                    checked={
                      patientData[
                        key as keyof Pick<
                          PatientData,
                          | "hypertension"
                          | "diabetes"
                          | "strokeHistory"
                          | "heartAttack"
                          | "bloodThinners"
                        >
                      ]
                    }
                    onChange={handleInputChange}
                    className="h-4 w-4 text-blue-500 rounded border-gray-300"
                  />
                  <span className="text-gray-700 capitalize">
                    {key.replace(/([A-Z])/g, " $1")}
                  </span>
                </label>
              ))}
            </div>

            <button
              onClick={handlePredict}
              disabled={loading}
              className="w-full mt-4 bg-green-500 text-white py-2 rounded hover:bg-green-600 transition disabled:opacity-50"
            >
              {loading ? "Analyzing..." : "Predict Treatment Outcome"}
            </button>
          </div>
        </div>

        {/* Right Column: Predictions */}
        <div className="space-y-6">
          {error && <div className="text-red-600 font-semibold">{error}</div>}

          {prediction && (
            <div className="space-y-6">
              <div className="bg-white shadow-md rounded-xl p-6 space-y-4">
                <h2 className="text-xl font-semibold text-gray-700">
                  Prediction Results
                </h2>

                <div className="bg-gray-50 p-4 rounded-lg shadow">
                  <h3 className="font-semibold text-gray-600 mb-2">
                    Treatment Recommendation
                  </h3>
                  <div
                    className={`font-bold text-white px-3 py-1 rounded mb-2 ${
                      prediction.prediction.predictedChange === "improve"
                        ? "bg-green-500"
                        : prediction.prediction.predictedChange === "decline"
                        ? "bg-red-500"
                        : "bg-yellow-500"
                    }`}
                  >
                    {prediction.prediction.recommendation}
                  </div>
                  <p>
                    Predicted change: {prediction.prediction.predictedChange}
                  </p>
                  <p>
                    Confidence:{" "}
                    {(prediction.prediction.confidence * 100).toFixed(1)}%
                  </p>
                  <p>Next follow-up: {prediction.prediction.nextFollowUp}</p>
                </div>

                <div className="bg-gray-50 p-4 rounded-lg shadow">
                  <h3 className="font-semibold text-gray-600 mb-2">
                    Visual Acuity
                  </h3>
                  <p>Current: {prediction.prediction.currentVisualAcuity}</p>
                  <p>
                    Predicted: {prediction.prediction.predictedVisualAcuity}
                  </p>
                  <p>
                    Change:{" "}
                    {(
                      prediction.prediction.predictedVisualAcuity -
                      prediction.prediction.currentVisualAcuity
                    ).toFixed(2)}
                  </p>
                </div>

                {prediction.charts?.main_chart && (
                  <div className="bg-gray-50 p-4 rounded-lg shadow">
                    <h3 className="font-semibold text-gray-600 mb-2">
                      Progression Chart
                    </h3>
                    <img
                      src={`data:image/png;base64,${prediction.charts.main_chart}`}
                      alt="Visual Acuity Progression"
                      className="rounded"
                    />
                  </div>
                )}

                <div className="bg-gray-50 p-4 rounded-lg shadow">
                  <h3 className="font-semibold text-gray-600 mb-2">
                    Statistics
                  </h3>
                  <p>Initial VA: {prediction.statistics.initialVisualAcuity}</p>
                  <p>Final VA: {prediction.statistics.finalVisualAcuity}</p>
                  <p>
                    Total injections: {prediction.statistics.totalInjections}
                  </p>
                </div>
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
