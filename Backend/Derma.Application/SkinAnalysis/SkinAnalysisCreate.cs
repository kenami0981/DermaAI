
using Derma.Infrastructure;
using MediatR;
using System;
using System.Threading;
using System.Threading.Tasks;

namespace Derma.Application.SkinAnalysis
{
    public class SkinAnalysisCreate
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

            public async Task<Unit> Handle(
                Command request,
                CancellationToken cancellationToken)
            {
                await _context.SkinAnalysis.AddAsync(
                    request.SkinAnalysis,
                    cancellationToken);

                await _context.SaveChangesAsync(cancellationToken);

                return Unit.Value;
            }
        }
    }
}
