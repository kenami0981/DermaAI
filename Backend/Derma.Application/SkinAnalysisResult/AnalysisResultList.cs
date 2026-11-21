using Derma.Infrastructure;
using MediatR;
using Microsoft.EntityFrameworkCore;

namespace Derma.Application.AnalysisResult
{
    public class AnalysisResultList
    {
        public class Query : IRequest<List<Derma.Domain.AnalysisResult>>
        {
            public Guid UserId { get; set; }
            public bool IsAdmin { get; set; }
        }

        public class Handler : IRequestHandler<Query, List<Derma.Domain.AnalysisResult>>
        {
            private readonly DataContext _context;

            public Handler(DataContext context)
            {
                _context = context;
            }

            public async Task<List<Derma.Domain.AnalysisResult>> Handle(Query request, CancellationToken cancellationToken)
            {
                var query = _context.AnalysisResult.Include(r => r.Detections).AsQueryable();

                if (!request.IsAdmin)
                    query = query.Where(r => _context.SkinAnalysis
                        .Any(a => a.Id == r.skinAnalysisId && a.userId == request.UserId));

                return await query.ToListAsync(cancellationToken);
            }
        }
    }
}