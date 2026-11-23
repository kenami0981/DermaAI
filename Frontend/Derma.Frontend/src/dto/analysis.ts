export enum AnalysisStatus {
    Pending = 0,
    Processing = 1,
    Completed = 2,
    Failed = 3,
}

export enum Tag {
    PoczatekCyklu = 0,
    SloneJedzenie = 1,
    OstreJedzenie = 2,
    FastFood = 3,
    Alkohol = 4,
}

export interface SkinAnalysisTag {
    id: string;
    skinAnalysisId: string;
    tag: Tag;
}

export interface Detection {
    id: string;
    analysisResultId: string;
    className: string;
    bbox: number[];
    confidence: number;
}

export interface AnalysisResult {
    id: string;
    skinAnalysisId: string;
    acneScore: number;
    acneCount: number;
    outputImageUrl: string | null;
    tta_applied: boolean | null;
    sahi_applied: boolean | null;
    detections: Detection[];
}

export interface SkinAnalysis {
    id: string;
    userId: string;
    notes: string | null;
    imageUrl: string;
    uploadDate: string;
    status: AnalysisStatus;
    tags: SkinAnalysisTag[];
    result: AnalysisResult | null;
}