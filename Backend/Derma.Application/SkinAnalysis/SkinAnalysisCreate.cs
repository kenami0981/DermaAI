
using Derma.Infrastructure;
using FluentValidation;
using MediatR;
using System;
using System.Security.Claims;
using System.Threading;
using System.Threading.Tasks;

namespace Derma.Application.SkinAnalysis
{
    public class SkinAnalysisCreate
    {
        public class CommandValidator : AbstractValidator<Command>
        {
            public CommandValidator()
            {
                RuleFor(x => x.SkinAnalysis).SetValidator(new SkinAnalysisValidator());
            }
        }
        public class Command : IRequest<Unit>
        {
            public required Derma.Domain.SkinAnalysis SkinAnalysis { get; set; }
        }

        public class Handler : IRequestHandler<Command, Unit>
        {
            private readonly DataContext _context;
            private readonly Microsoft.AspNetCore.Http.IHttpContextAccessor _httpContextAccessor;

            public Handler(DataContext context, Microsoft.AspNetCore.Http.IHttpContextAccessor httpContextAccessor)
            {
                _context = context;
                _httpContextAccessor = httpContextAccessor;
            }

            public async Task<Unit> Handle(
                Command request,
                CancellationToken cancellationToken)
            {


                var userId = _httpContextAccessor.HttpContext?
                    .User.FindFirst(ClaimTypes.NameIdentifier)?.Value;

                if (!Guid.TryParse(userId, out var id))
                    throw new UnauthorizedAccessException("Nieprawidłowy UserId.");

                var skinAnalysis = request.SkinAnalysis;

                skinAnalysis.Id = Guid.NewGuid();
                skinAnalysis.userId = id;

                await _context.SkinAnalysis.AddAsync(
                    skinAnalysis,
                    cancellationToken
                );

                await _context.SaveChangesAsync(cancellationToken);

                return Unit.Value;
            }
        }
    }
}
