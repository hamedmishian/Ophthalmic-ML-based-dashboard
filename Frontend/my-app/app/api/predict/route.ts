import { type NextRequest, NextResponse } from "next/server"

export async function POST(request: NextRequest) {
  try {
    const { patientInfo } = await request.json()

    // Simulate processing delay
    await new Promise((resolve) => setTimeout(resolve, 2000))

    // Mock prediction logic based on patient data
    const age = new Date().getFullYear() - patientInfo.birthYear
    const riskFactors = [
      patientInfo.hypertension,
      patientInfo.diabetes,
      patientInfo.strokeHistory,
      patientInfo.heartAttack,
    ].filter(Boolean).length

    // Simulate prediction based on risk factors and age
    const baseConfidence = 0.75 + riskFactors * 0.05
    const confidence = Math.min(0.95, baseConfidence)

    let predictedChange = "stable"
    let recommendation = "Continue current treatment"

    if (patientInfo.initialVisualAcuity < 0.2 || riskFactors >= 3) {
      predictedChange = "decline"
      recommendation = "Increase injection frequency"
    } else if (patientInfo.initialVisualAcuity > 0.5 && riskFactors <= 1) {
      predictedChange = "improve"
      recommendation = "Maintain current treatment"
    }

    const currentVA = patientInfo.initialVisualAcuity
    const predictedVA =
      predictedChange === "improve" ? currentVA + 0.1 : predictedChange === "decline" ? currentVA - 0.05 : currentVA

    // Generate a simple base64 chart placeholder
    const chartData = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNkYPhfDwAChwGA60e6kgAAAABJRU5ErkJggg=="

    const mockResponse = {
      success: true,
      prediction: {
        recommendation,
        predictedChange,
        confidence,
        nextFollowUp: "4 weeks",
        currentVisualAcuity: currentVA,
        predictedVisualAcuity: Math.round(predictedVA * 100) / 100,
      },
      charts: {
        main_chart: chartData,
      },
      statistics: {
        initialVisualAcuity: currentVA,
        finalVisualAcuity: Math.round(predictedVA * 100) / 100,
        totalInjections: Math.floor(Math.random() * 12) + 1,
      },
    }

    return NextResponse.json(mockResponse)
  } catch (error) {
    return NextResponse.json({ success: false, error: "Prediction failed" }, { status: 500 })
  }
}
