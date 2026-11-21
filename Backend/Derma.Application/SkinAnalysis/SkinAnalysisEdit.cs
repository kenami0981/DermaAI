using Derma.Domain;
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
    public class SkinAnalysisEdit
    {
        public class Command : IRequest<bool>
        {
            public required Derma.Domain.SkinAnalysis SkinAnalysis { get; set; }
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
                var query = _context.SkinAnalysis
                    .Include(a => a.Tags)
                    .Where(a => a.Id == request.SkinAnalysis.Id);

                if (!request.IsAdmin)
                    query = query.Where(a => a.userId == request.UserId);

                var analysis = await query.FirstOrDefaultAsync(cancellationToken);
                if (analysis == null) return false;

                analysis.notes = request.SkinAnalysis.notes;
                analysis.uploadDate = request.SkinAnalysis.uploadDate;

                var wanted = request.SkinAnalysis.Tags.Select(t => t.Tag).Distinct().ToList();

                var toRemove = analysis.Tags.Where(t => !wanted.Contains(t.Tag)).ToList();
                _context.SkinAnalysisTag.RemoveRange(toRemove);

                var existing = analysis.Tags.Select(t => t.Tag).ToList();
                foreach (var tag in wanted.Except(existing))
                    analysis.Tags.Add(new SkinAnalysisTag { Id = Guid.NewGuid(), Tag = tag });

                await _context.SaveChangesAsync(cancellationToken);
                return true;
            }
        }
    }
}
