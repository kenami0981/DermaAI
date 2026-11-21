using Derma.Infrastructure;
using MediatR;
using Microsoft.EntityFrameworkCore;

namespace Derma.Application.AnalysisResult
{
    public class AnalysisResultDetails
    {
        public class Query : IRequest<Derma.Domain.AnalysisResult?>
        {
            public Guid Id { get; set; }
            public Guid UserId { get; set; }
            public bool IsAdmin { get; set; }
        }

        public class Handler : IRequestHandler<Query, Derma.Domain.AnalysisResult?>
        {
            private readonly DataContext _context;

            public Handler(DataContext context)
            {
                _context = context;
            }

            public async Task<Derma.Domain.AnalysisResult?> Handle(Query request, CancellationToken cancellationToken)
            {
                var query = _context.AnalysisResult.Include(r => r.Detections).Where(r => r.Id == request.Id);

                if (!request.IsAdmin)
                    query = query.Where(r => _context.SkinAnalysis
                        .Any(a => a.Id == r.skinAnalysisId && a.userId == request.UserId));

                return await query.FirstOrDefaultAsync(cancellationToken);
            }
        }
    }
}