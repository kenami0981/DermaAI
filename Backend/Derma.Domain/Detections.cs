using System;
using System.Collections.Generic;
using System.Data;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Derma.Domain
{

    public class Detection {
        public Guid Id { get; set; }
        public Guid AnalysisResultId { get; set; }
        public string className { get; set; }
        public double[] bbox { get; set; } // [12.32,213.32,132.23,23.0]
        public double confidence { get; set; }

    }
    public class AnalysisResult

    {
        public Guid Id { get; set; }
        public Guid skinAnalysisId { get; set; }
        public double acneScore { get; set; }
        public int acneCount { get; set; }
        public string? outputImageUrl { get; set; }
        public bool? tta_applied { get; set; }
        public bool? sahi_applied { get; set; }
        public ICollection<Detection> Detections { get; set; } = new List<Detection>();


    }
}
