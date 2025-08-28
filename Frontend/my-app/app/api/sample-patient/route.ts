import { NextResponse } from "next/server"

export async function GET() {
  const samplePatient = {
    patientInfo: {
      patientId: "AMD-2024-001",
      birthYear: 1955,
      gender: "female",
      hypertension: true,
      diabetes: false,
      strokeHistory: false,
      heartAttack: false,
      bloodThinners: true,
      initialVisualAcuity: 0.4,
      initialThickness: 380,
    },
  }

  return NextResponse.json(samplePatient)
}
