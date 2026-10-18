using Derma.Infrastructure;
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

        public class Handler : IRequestHandler<Command, Guid>
        {
            private readonly DataContext _context;

            public Handler(DataContext context)
            {
                _context = context;
            }

            public async Task<Guid> Handle(Command request, CancellationToken cancellationToken)
            {
                var cosmeticUsage = request.CosmeticUsage;

                cosmeticUsage.Id = Guid.NewGuid();

                _context.CosmeticUsage.Add(cosmeticUsage);
                await _context.SaveChangesAsync(cancellationToken);

                return cosmeticUsage.Id;
            }
        }
    }
}
