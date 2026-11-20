using Derma.Infrastructure;
using FluentValidation;
using MediatR;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Derma.Application.CosmeticUsage
{
    public class CosmeticUsageCreate
    {
        public class Command : IRequest<Guid>
        {
            public required Derma.Domain.CosmeticUsage CosmeticUsage { get; set; }
        }
        public class CommandValidator : AbstractValidator<Command>
        {
            public CommandValidator()
            {
                RuleFor(x => x.CosmeticUsage).SetValidator(new CosmeticUsageValidator());
            }
        }

        public class Handler : IRequestHandler<Command, Guid>
        { 

            private readonly DataContext _context;
            private readonly Microsoft.AspNetCore.Http.IHttpContextAccessor _httpContextAccessor;

            public Handler(DataContext context, Microsoft.AspNetCore.Http.IHttpContextAccessor httpContextAccessor)
            {
                _context = context;
                _httpContextAccessor = httpContextAccessor;
            }


            public async Task<Guid> Handle(Command request, CancellationToken cancellationToken)
            {
                var cosmeticUsage = request.CosmeticUsage;

                var userId = _httpContextAccessor.HttpContext?
    .User.FindFirst(System.Security.Claims.ClaimTypes.NameIdentifier)?.Value;

                if (!Guid.TryParse(userId, out var id))
                    throw new UnauthorizedAccessException("Nieprawidłowy UserId.");

                cosmeticUsage.Id = Guid.NewGuid();
                cosmeticUsage.userId = id;

                _context.CosmeticUsage.Add(cosmeticUsage);
                await _context.SaveChangesAsync(cancellationToken);

                return cosmeticUsage.Id;
            }
        }
    }
}
