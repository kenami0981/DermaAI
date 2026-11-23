using Derma.Infrastructure;
using MediatR;
using Microsoft.EntityFrameworkCore;

namespace Derma.Application.AnalysisResult
{
    public class AnalysisResultDelete
    {
        public class Command : IRequest<bool>
        {
            public Guid Id { get; set; }
            public Guid UserId { get; set; }
            public bool IsAdmin { get; set; }
        }

        public class Handler : IRequestHandler<Command, bool>
        {
            private readonly DataContext _context;

            public Handler(DataContext context)
            {
                _context = context;
            }

            public async Task<bool> Handle(Command request, CancellationToken cancellationToken)
            {
                var query = _context.AnalysisResult.Where(r => r.Id == request.Id);

                if (!request.IsAdmin)
                    query = query.Where(r => _context.SkinAnalysis
                        .Any(a => a.Id == r.skinAnalysisId && a.userId == request.UserId));

                var result = await query.FirstOrDefaultAsync(cancellationToken);
                if (result == null) return false;

                _context.AnalysisResult.Remove(result);
                await _context.SaveChangesAsync(cancellationToken);
                return true;
            }
        }
    }
}