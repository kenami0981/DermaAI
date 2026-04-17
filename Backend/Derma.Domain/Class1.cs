namespace Derma.Domain
{
    public class SkinAnalysis
    {
        public Guid Id { get; set; }
        public Guid userId { get; set; }
        public string? notes { get; set; }
        public string imageUrl {  get; set; }
        public DateTime uploadDate { get; set; }
        public string? acneLevel { get; set; }
        public double? confidence { get; set; }
    }
}
