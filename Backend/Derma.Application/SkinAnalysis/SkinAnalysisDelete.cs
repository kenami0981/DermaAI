
using Derma.Infrastructure;
using MediatR;
using Microsoft.EntityFrameworkCore;
using System;
using System.Threading;
using System.Threading.Tasks;

namespace Derma.Application.SkinAnalysis
{
    public class SkinAnalysisDelete
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

            public async Task<bool> Handle(
                Command request,
                CancellationToken cancellationToken)
            {
                var query = _context.SkinAnalysis.Where(a => a.Id == request.Id);
                if (!request.IsAdmin)
                    query = query.Where(a => a.userId == request.UserId);
                var skinAnalysis = await query.FirstOrDefaultAsync(cancellationToken);

                if (skinAnalysis == null) return false;
                _context.SkinAnalysis.Remove(skinAnalysis);

                await _context.SaveChangesAsync(cancellationToken);
                

                return true;
            }
        }
    }
}
