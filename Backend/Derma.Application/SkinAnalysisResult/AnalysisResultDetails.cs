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
                return await _context.AnalysisResult
                    .Include(r => r.Detections)
                    .FirstOrDefaultAsync(r => r.Id == request.Id, cancellationToken);
            }
        }
    }
}