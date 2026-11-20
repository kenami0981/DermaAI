using FluentValidation;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Derma.Application.SkinAnalysis
{
    public class SkinAnalysisValidator : AbstractValidator<Derma.Domain.SkinAnalysis>
    {
        public SkinAnalysisValidator()
        {
            RuleFor(x => x.notes)
                .MaximumLength(1000)
                .WithMessage("Notatka może mieć maksymalnie 1000 znaków.");
            RuleFor(x => x.uploadDate)
                .NotEmpty()
                .WithMessage("UploadDate jest wymagana.")
                .Must(date => date <= DateTime.UtcNow.AddMinutes(5))
                .WithMessage("UploadDate nie może być datą z przyszłości.");
            RuleForEach(x => x.Tags).ChildRules(tag =>
            {
                tag.RuleFor(t => t.Tag).IsInEnum().WithMessage("Nieprawidłowy typ tagu.");
            });
        }

    }
}
