using Derma.Infrastructure;
using MediatR;

namespace Derma.Application.AnalysisResult
{
    public class AnalysisResultDelete
    {
        public class Command : IRequest<Unit>
        {
            public Guid Id { get; set; }
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
                var analysisResult = await _context.AnalysisResult.FindAsync(request.Id);

                _context.AnalysisResult.Remove(analysisResult);
                await _context.SaveChangesAsync(cancellationToken);

                return Unit.Value;
            }
        }
    }
}