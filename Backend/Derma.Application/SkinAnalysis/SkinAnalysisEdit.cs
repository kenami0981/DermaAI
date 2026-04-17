using Derma.Infrastructure;
using MediatR;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Derma.Application.SkinAnalysis
{
    public class SkinAnalysisEdit
    {
        public class Command : IRequest<Unit>
        {
            public required Derma.Domain.SkinAnalysis SkinAnalysis { get; set; }
        }
        public class Handler : IRequestHandler<Command, Unit>
        {
            private readonly DataContext _context;
            public Handler(DataContext context)
            {
                _context = context;
            }
            public async Task<Unit> Handle(Command request, CancellationToken cancellationToken)
            {
                var skinAnalysis = await _context.SkinAnalysis.FindAsync(request.SkinAnalysis.Id);
                skinAnalysis.imageUrl = request.SkinAnalysis.imageUrl;
                skinAnalysis.notes = request.SkinAnalysis.notes ?? skinAnalysis.notes;
                skinAnalysis.acneLevel = request.SkinAnalysis.acneLevel;
                skinAnalysis.confidence = request.SkinAnalysis.confidence ?? skinAnalysis.confidence;
                skinAnalysis.uploadDate = request.SkinAnalysis.uploadDate;
                await _context.SaveChangesAsync();

                return Unit.Value;

            }
        }
    }
}
