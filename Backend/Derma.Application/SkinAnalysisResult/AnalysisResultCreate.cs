using Derma.Infrastructure;
using MediatR;

namespace Derma.Application.AnalysisResult
{
    public class AnalysisResultCreate
    {
        public class Command : IRequest<Unit>
        {
            public required Derma.Domain.AnalysisResult AnalysisResult { get; set; }
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
                _context.AnalysisResult.Add(request.AnalysisResult);
                await _context.SaveChangesAsync(cancellationToken);

                return Unit.Value;
            }
        }
    }
}