using System.ComponentModel.DataAnnotations;

namespace Derma.Domain
{
    public enum AnalysisStatus { Pending, Processing, Completed, Failed }
    public enum Tag
    {
        PoczatekCyklu,
        SloneJedzenie,
        OstreJedzenie,
        FastFood,
        Alkohol
    }
    public class SkinAnalysisTag
    {
        public Guid Id { get; set; }
        public Guid SkinAnalysisId { get; set; }
        public Tag Tag { get; set; }
        public bool Value { get; set; }
    }
    public class SkinAnalysis
    {
        public Guid Id { get; set; }
        public Guid userId { get; set; }
        public string? notes { get; set; }
        public string imageUrl {  get; set; }
        public DateTime uploadDate { get; set; }
        public AnalysisStatus Status { get; set; }
        public ICollection<SkinAnalysisTag> Tags { get; set; } = new List<SkinAnalysisTag>();
        public AnalysisResult? Result { get; set; }
    }
}
