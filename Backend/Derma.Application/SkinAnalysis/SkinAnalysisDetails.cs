using Derma.Infrastructure;
using MediatR;
using Microsoft.EntityFrameworkCore;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Derma.Application.SkinAnalysis
{
    public class SkinAnalysisDetails
    {
        public class Query : IRequest<Derma.Domain.SkinAnalysis?>
        {
            public Guid Id { get; set; }
            public Guid UserId { get; set; }
            public bool IsAdmin { get; set; }
        }


        public class Handler : IRequestHandler<Query, Derma.Domain.SkinAnalysis>
        {
            private readonly DataContext _context;
            public Handler(DataContext context)
            {
                _context = context;
            }
            public async Task<Derma.Domain.SkinAnalysis?> Handle(Query request, CancellationToken cancellationToken)
            {
                var query = _context.SkinAnalysis
                    .Include(a => a.Tags)
                    .Include(a => a.Result)
                    .Where(a => a.Id == request.Id);

                if (!request.IsAdmin)
                    query = query.Where(a => a.userId == request.UserId);

                return await query.FirstOrDefaultAsync(cancellationToken);
            }
        }
    }
}

