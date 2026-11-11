namespace Derma.API.Models
{
    public class SkinAnalysisCreateRequest
    {
        public required IFormFile Image { get; set; }
        public string? notes { get; set; }
        public DateTime uploadDate { get; set; }
        public List<Derma.Domain.Tag>? tags { get; set; }
    }

    public class TagRequest
    {
        public Derma.Domain.Tag tag { get; set; }
        public bool value { get; set; }
    }
}