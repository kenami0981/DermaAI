using Derma.Infrastructure;
using MediatR;
using Microsoft.EntityFrameworkCore;

namespace Derma.Application.AnalysisResult
{
    public class AnalysisResultList
    {
        public class Query : IRequest<List<Derma.Domain.AnalysisResult>> { }

        public class Handler : IRequestHandler<Query, List<Derma.Domain.AnalysisResult>>
        {
            private readonly DataContext _context;

            public Handler(DataContext context)
            {
                _context = context;
            }

            public async Task<List<Derma.Domain.AnalysisResult>> Handle(Query request, CancellationToken cancellationToken)
            {
                return await _context.AnalysisResult
                    .Include(r => r.Detections)
                    .ToListAsync(cancellationToken);
            }
        }
    }
}